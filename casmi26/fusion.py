"""Turn per-spectrum hits into one ranked candidate list per molecule.

The metric scores molecules, not spectra. A molecule arrives as 1-16 spectra
taken at different collision energies and as different adducts, and they carry
genuinely different information: a low-energy spectrum barely fragments, a
high-energy one is all small pieces. Agreement between them is the single most
useful confidence signal available, which is why it is computed here and fed
straight into the gate.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .library import Hit, SpectralLibrary


@dataclass
class Candidate:
    key: str
    smiles: str
    best_score: float = 0.0
    sum_score: float = 0.0
    n_spectra_supporting: int = 0
    best_rank: int = 10 ** 9
    rrf: float = 0.0          # reciprocal-rank fusion across spectra
    channels: set = field(default_factory=set)


def fuse_hits(hits_per_spectrum: list[list[Hit]], library: SpectralLibrary,
              rrf_k: int = 60, max_hits_per_spectrum: int = 100) -> list[Candidate]:
    """Collapse hits from every spectrum of one molecule onto structures.

    Reciprocal-rank fusion is the primary ordering because raw cosine is not
    comparable across spectra -- a clean high-energy spectrum scores higher
    against everything than a sparse low-energy one, so summing raw scores lets
    one spectrum dominate the vote.
    """
    by_key: dict[str, Candidate] = {}
    for hits in hits_per_spectrum:
        seen_this_spectrum: set[str] = set()
        for rank, h in enumerate(hits[:max_hits_per_spectrum], start=1):
            key = str(library.keys[h.library_row])
            cand = by_key.get(key)
            if cand is None:
                cand = Candidate(key=key, smiles=str(library.smiles[h.library_row]))
                by_key[key] = cand
            cand.channels.add(h.channel)
            if h.score > cand.best_score:
                cand.best_score = h.score
            cand.best_rank = min(cand.best_rank, rank)
            if key not in seen_this_spectrum:
                # count each spectrum once even if the structure appears in it twice
                cand.sum_score += h.score
                cand.rrf += 1.0 / (rrf_k + rank)
                cand.n_spectra_supporting += 1
                seen_this_spectrum.add(key)
    out = list(by_key.values())
    out.sort(key=lambda c: (-c.rrf, -c.best_score))
    return out


def gate_features(candidates: list[Candidate], n_spectra: int,
                  query_meta: dict | None = None) -> dict[str, float]:
    """Features describing HOW CONFIDENT the library answer is.

    This is not about which structure is right. It is about whether the top
    library candidate deserves rank 1, or whether retrieved candidates should
    take the top slots instead. Everything here is cheap and available before
    any model is trained.
    """
    query_meta = query_meta or {}
    f: dict[str, float] = {}
    top = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None

    f["n_candidates"] = float(len(candidates))
    f["n_spectra"] = float(n_spectra)
    f["top_best_score"] = top.best_score if top else 0.0
    f["top_rrf"] = top.rrf if top else 0.0
    f["top_sum_score"] = top.sum_score if top else 0.0
    f["top_support_frac"] = (top.n_spectra_supporting / n_spectra) if (top and n_spectra) else 0.0
    f["top_is_fragment_channel"] = float("fragment" in top.channels) if top else 0.0

    # Margin: how far clear of the runner-up is the top candidate. A big margin
    # with several spectra agreeing is the signature of a real class-1 hit.
    f["margin_score"] = (top.best_score - second.best_score) if (top and second) else (top.best_score if top else 0.0)
    f["margin_rrf"] = (top.rrf - second.rrf) if (top and second) else (top.rrf if top else 0.0)
    f["margin_ratio"] = (top.best_score / second.best_score) if (top and second and second.best_score > 0) else 10.0

    scores = np.array([c.best_score for c in candidates[:50]], dtype=np.float64)
    if scores.size:
        f["score_mean"] = float(scores.mean())
        f["score_std"] = float(scores.std())
        f["score_top_minus_mean"] = float(scores[0] - scores.mean())
        f["score_entropy"] = _entropy(scores)
    else:
        f["score_mean"] = f["score_std"] = f["score_top_minus_mean"] = f["score_entropy"] = 0.0

    f["precursor_mz"] = float(query_meta.get("precursor_mz", 0.0))
    f["mean_n_peaks"] = float(query_meta.get("mean_n_peaks", 0.0))
    f["log_base_peak"] = float(np.log1p(query_meta.get("base_peak_intensity", 0.0) or 0.0))
    f["n_adducts"] = float(query_meta.get("n_adducts", 1))
    f["n_energies"] = float(query_meta.get("n_energies", 1))
    return f


def _entropy(scores: np.ndarray) -> float:
    """Entropy of the normalised score distribution.

    Low entropy = one candidate dominates = confident. High entropy = the
    library is shrugging, and that is exactly when retrieval should take over.
    """
    s = np.clip(scores, 1e-12, None)
    p = s / s.sum()
    return float(-(p * np.log(p)).sum())


def rank_candidates(candidates: list[Candidate], k: int = 25) -> list[str]:
    """SMILES of the top-k candidates, best first."""
    return [c.smiles for c in candidates[:k]]
