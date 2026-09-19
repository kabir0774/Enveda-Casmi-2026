"""Spectral library search -- the class-1 workhorse.

Design notes:

* Spectra are L2-normalised sparse vectors, so cosine similarity is one sparse
  matrix product. On 2.5M library spectra this is the only formulation that
  finishes in a Kaggle notebook.
* Two channels are searched: fragment m/z and neutral loss. The neutral-loss
  channel is what finds analogues -- a molecule that differs from a library
  entry by one substituent keeps its losses.
* Queries are blocked so peak memory stays bounded regardless of library size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .preprocess import BinConfig, CleanConfig, clean_peaks, neutral_loss_bins, stack_to_csr, to_bins


@dataclass
class Hit:
    library_row: int
    score: float
    channel: str


class SpectralLibrary:
    """Reference spectra with known structures, searchable by cosine.

    `keys` holds one identity per library row (InChIKey14 in practice). Rows
    sharing a key are the same structure measured more than once.
    """

    def __init__(self, keys: np.ndarray, smiles: np.ndarray,
                 precursor_mz: np.ndarray,
                 frag: sparse.csr_matrix, loss: sparse.csr_matrix):
        assert frag.shape[0] == loss.shape[0] == len(keys) == len(smiles)
        self.keys = np.asarray(keys)
        self.smiles = np.asarray(smiles)
        self.precursor_mz = np.asarray(precursor_mz, dtype=np.float64)
        self.frag = frag.T.tocsr()   # transposed once: (n_bins, n_lib)
        self.loss = loss.T.tocsr()
        self.n = len(keys)

    @classmethod
    def build(cls, spectra: list[dict], bin_cfg: BinConfig = BinConfig(),
              clean_cfg: CleanConfig = CleanConfig()) -> "SpectralLibrary":
        """spectra: dicts with mzs, intensities, precursor_mz, smiles, key."""
        frag_rows, loss_rows, keys, smis, prec = [], [], [], [], []
        for s in spectra:
            mz, it = clean_peaks(s["mzs"], s["intensities"], s["precursor_mz"], clean_cfg)
            if mz.size == 0:
                continue
            frag_rows.append(to_bins(mz, it, bin_cfg))
            loss_rows.append(neutral_loss_bins(mz, it, s["precursor_mz"], bin_cfg))
            keys.append(s["key"])
            smis.append(s["smiles"])
            prec.append(s["precursor_mz"])
        frag = stack_to_csr(frag_rows, bin_cfg.n_bins)
        loss = stack_to_csr(loss_rows, bin_cfg.n_bins)
        return cls(np.array(keys), np.array(smis), np.array(prec), frag, loss)

    def search(self, queries: list[dict], top_k: int = 200,
               bin_cfg: BinConfig = BinConfig(), clean_cfg: CleanConfig = CleanConfig(),
               loss_weight: float = 0.5, block: int | None = None,
               precursor_tol: float | None = None,
               max_block_bytes: int = 512 * 1024 * 1024) -> list[list[Hit]]:
        """Top-k library rows per query spectrum, fragment + loss channels fused.

        `precursor_tol` (in Da) restricts hits to library entries of nearly the
        same precursor mass -- that is the exact-match regime. Leave it None
        for analogue search, where the mass is allowed to differ.
        """
        frag_rows, loss_rows, qprec = [], [], []
        for q in queries:
            mz, it = clean_peaks(q["mzs"], q["intensities"], q["precursor_mz"], clean_cfg)
            frag_rows.append(to_bins(mz, it, bin_cfg))
            loss_rows.append(neutral_loss_bins(mz, it, q["precursor_mz"], bin_cfg))
            qprec.append(q["precursor_mz"])
        Q_frag = stack_to_csr(frag_rows, bin_cfg.n_bins)
        Q_loss = stack_to_csr(loss_rows, bin_cfg.n_bins)
        qprec = np.asarray(qprec, dtype=np.float64)

        # The score block is dense: block x n_library float64. At 1.9M library
        # spectra a block of 256 is 4 GB per channel, which is how a search that
        # worked at 100k library spectra kills the machine at full scale.
        if block is None:
            per_row = self.n * 8 * 2  # two channels, float64
            block = max(1, min(256, max_block_bytes // max(per_row, 1)))

        out: list[list[Hit]] = []
        for start in range(0, Q_frag.shape[0], block):
            stop = min(start + block, Q_frag.shape[0])
            sf = (Q_frag[start:stop] @ self.frag).toarray()
            sl = (Q_loss[start:stop] @ self.loss).toarray()
            combined = sf + loss_weight * sl
            if precursor_tol is not None:
                mask = np.abs(self.precursor_mz[None, :] - qprec[start:stop, None]) > precursor_tol
                combined[mask] = -np.inf
            for r in range(combined.shape[0]):
                row = combined[r]
                k = min(top_k, row.size)
                idx = np.argpartition(row, -k)[-k:]
                idx = idx[np.argsort(row[idx])[::-1]]
                hits = []
                for j in idx:
                    if not np.isfinite(row[j]) or row[j] <= 0:
                        continue
                    channel = "fragment" if sf[r, j] >= loss_weight * sl[r, j] else "loss"
                    hits.append(Hit(int(j), float(row[j]), channel))
                out.append(hits)
        return out
