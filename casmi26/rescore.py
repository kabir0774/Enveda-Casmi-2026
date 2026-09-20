"""Rerank a candidate shortlist with entropy similarity.

Measured on real data: cosine search puts the correct structure somewhere in
the candidate list 86% of the time and at rank 1 only 55% of the time. So the
retrieval stage is close to its ceiling and the ordering is not. Reranking the
shortlist is the cheapest place to attack that, because it touches ~150
candidates per molecule rather than 500,000 library spectra.
"""
from __future__ import annotations

import numpy as np

from .fusion import Candidate
from .library import SpectralLibrary
from .similarity import entropy_similarity, neutral_loss_similarity


def rescore_candidates(candidates: list[Candidate], query_spectra: list[dict],
                       library: SpectralLibrary, top_n: int = 100,
                       tol: float = 0.02, loss_weight: float = 0.5,
                       blend: float = 1.0) -> list[Candidate]:
    """Reorder the top_n candidates by entropy similarity to the query.

    `blend` mixes the new score with the original fusion rank: 1.0 uses entropy
    alone, 0.0 leaves the order untouched. Anything below the top_n cut keeps
    its original position, so this can only reorder the head of the list.
    """
    if library.peaks is None or not candidates:
        return candidates

    head, tail = candidates[:top_n], candidates[top_n:]
    scored = []
    for rank0, cand in enumerate(head, start=1):
        if cand.best_row < 0 or cand.best_row >= len(library.peaks):
            scored.append((cand, 0.0, rank0))
            continue
        lib_mz, lib_int = library.peaks[cand.best_row]
        lib_prec = float(library.precursor_mz[cand.best_row])
        best = 0.0
        for q in query_spectra:
            frag = entropy_similarity(q["_mzs"], q["_ints"], lib_mz, lib_int, tol)
            loss = neutral_loss_similarity(q["_mzs"], q["_ints"], q["precursor_mz"],
                                           lib_mz, lib_int, lib_prec, tol)
            best = max(best, frag + loss_weight * loss)
        scored.append((cand, best, rank0))

    # Blend the entropy score with the original ordering. Reciprocal rank keeps
    # the two on a comparable scale without needing the raw cosine values.
    def key(item):
        cand, ent, rank0 = item
        return -(blend * ent + (1.0 - blend) * (1.0 / rank0))

    scored.sort(key=key)
    return [c for c, _, _ in scored] + tail
