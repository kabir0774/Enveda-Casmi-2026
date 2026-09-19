"""Spectrum cleaning and vector encoding.

Two representations come out of here:

  fragment vector  -- binned m/z of the peaks as measured
  neutral-loss vector -- binned (precursor - m/z)

The neutral-loss view matters because two analogues that differ by one
substituent share their losses even when their fragment masses are all shifted.
That is most of what "modified cosine" buys you, without the O(n*m) peak
matching.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

PROTON = 1.007276
C13_C12 = 1.003355


@dataclass(frozen=True)
class CleanConfig:
    min_rel_intensity: float = 0.01      # drop peaks under 1% of base peak
    precursor_margin: float = 2.0        # drop peaks above precursor + this
    remove_precursor_window: float = 0.0 # set >0 to also drop the precursor itself
    deisotope_tol: float = 0.01
    window_da: float = 50.0              # top-n peaks per this-wide window
    peaks_per_window: int = 6
    max_peaks: int = 128
    min_peaks: int = 3
    intensity_power: float = 0.5         # 0.5 = sqrt, 1.0 = raw, 0.0 = binary


@dataclass(frozen=True)
class BinConfig:
    bin_size: float = 0.05
    max_mz: float = 1200.0
    mz_power: float = 0.0
    """Weight each peak by (m/z) ** mz_power before scoring.

    Plain cosine treats a 91 Da tropylium fragment -- which half of all
    aromatics produce -- as being as informative as a 437 Da fragment that
    almost uniquely identifies a scaffold. Heavy fragments carry far more
    structural information, and up-weighting them is standard in library
    search (NIST-style weighting uses roughly mz**2 with sqrt intensities).
    0.0 reproduces plain cosine.
    """

    @property
    def n_bins(self) -> int:
        return int(np.ceil(self.max_mz / self.bin_size)) + 1


def clean_peaks(mzs, intensities, precursor_mz: float,
                cfg: CleanConfig = CleanConfig()):
    """Return (mzs, intensities) after the standard curation steps.

    Order matters: intensity floor, then precursor cut, then deisotope, then
    windowed top-n, then the intensity transform. Doing top-n before the
    floor keeps noise that the floor would have removed.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(intensities, dtype=np.float64)
    if mzs.size == 0:
        return mzs, ints

    order = np.argsort(mzs)
    mzs, ints = mzs[order], ints[order]

    base = ints.max()
    if base <= 0:
        return mzs[:0], ints[:0]
    keep = ints >= cfg.min_rel_intensity * base
    mzs, ints = mzs[keep], ints[keep]

    keep = mzs <= precursor_mz + cfg.precursor_margin
    mzs, ints = mzs[keep], ints[keep]
    if cfg.remove_precursor_window > 0:
        keep = np.abs(mzs - precursor_mz) > cfg.remove_precursor_window
        mzs, ints = mzs[keep], ints[keep]

    if cfg.deisotope_tol > 0 and mzs.size > 1:
        mzs, ints = _deisotope(mzs, ints, cfg.deisotope_tol)

    if cfg.peaks_per_window > 0:
        mzs, ints = _window_top_n(mzs, ints, cfg.window_da, cfg.peaks_per_window)

    if mzs.size > cfg.max_peaks:
        idx = np.argsort(ints)[::-1][: cfg.max_peaks]
        idx.sort()
        mzs, ints = mzs[idx], ints[idx]

    if mzs.size == 0:
        return mzs, ints
    if cfg.intensity_power == 0.0:
        ints = np.ones_like(ints)
    elif cfg.intensity_power != 1.0:
        ints = np.power(ints, cfg.intensity_power)
    m = ints.max()
    if m > 0:
        ints = ints / m
    return mzs, ints


def _deisotope(mzs, ints, tol):
    """Drop a peak that sits ~1.0034 Da above a more intense peak.

    Vectorised: the per-peak isotope window is found with one searchsorted over
    the whole array, and only the few peaks that actually have a partner enter
    a Python loop. On real spectra with hundreds of peaks this is the
    difference between preprocessing being free and it dominating training.
    """
    n = mzs.size
    if n < 2:
        return mzs, ints
    targets = mzs + C13_C12
    lo = np.searchsorted(mzs, targets - tol)
    hi = np.searchsorted(mzs, targets + tol)
    has_partner = np.flatnonzero(hi > lo)
    if has_partner.size == 0:
        return mzs, ints
    keep = np.ones(n, dtype=bool)
    for i in has_partner:
        if not keep[i]:
            continue
        for j in range(lo[i], hi[i]):
            if j != i and ints[j] <= ints[i]:
                keep[j] = False
    return mzs[keep], ints[keep]


def _window_top_n(mzs, ints, window, n):
    """Keep the n most intense peaks in each window-wide slice of m/z.

    Preserves informative low-mass fragments that a global top-N throws away
    because they are dim next to the base peak.

    Vectorised with a lexsort: peaks are ordered by (window, -intensity) and
    each peak's position within its own window is read off a cumulative count,
    so no per-window Python loop is needed.
    """
    if mzs.size == 0:
        return mzs, ints
    bucket = np.floor(mzs / window).astype(np.int64)
    order = np.lexsort((-ints, bucket))
    sorted_bucket = bucket[order]
    # position of each element within its bucket, after sorting by intensity
    starts = np.flatnonzero(np.r_[True, sorted_bucket[1:] != sorted_bucket[:-1]])
    within = np.arange(sorted_bucket.size) - np.repeat(starts, np.diff(np.r_[starts, sorted_bucket.size]))
    keep = np.zeros(mzs.size, dtype=bool)
    keep[order[within < n]] = True
    return mzs[keep], ints[keep]


def to_bins(mzs, ints, cfg: BinConfig = BinConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Bin peaks onto a fixed grid, summing intensity inside a bin."""
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    keep = (mzs >= 0) & (mzs <= cfg.max_mz)
    mzs, ints = mzs[keep], ints[keep]
    if mzs.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    if cfg.mz_power:
        ints = ints * np.power(np.clip(mzs, 1.0, None), cfg.mz_power)
    idx = np.rint(mzs / cfg.bin_size).astype(np.int32)
    order = np.argsort(idx)
    idx, vals = idx[order], ints[order]
    uniq, start = np.unique(idx, return_index=True)
    summed = np.add.reduceat(vals, start)
    return uniq.astype(np.int32), summed.astype(np.float32)


def neutral_loss_bins(mzs, ints, precursor_mz: float,
                      cfg: BinConfig = BinConfig()) -> tuple[np.ndarray, np.ndarray]:
    """Bin the losses (precursor - fragment) instead of the fragments.

    Weighting is applied on the FRAGMENT mass, not the loss: a small loss from
    a large fragment is informative, and weighting by the loss would penalise
    exactly those.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if cfg.mz_power:
        ints = ints * np.power(np.clip(mzs, 1.0, None), cfg.mz_power)
    losses = float(precursor_mz) - mzs
    keep = losses >= 0
    flat = BinConfig(bin_size=cfg.bin_size, max_mz=cfg.max_mz, mz_power=0.0)
    return to_bins(losses[keep], ints[keep], flat)


def stack_to_csr(rows: list[tuple[np.ndarray, np.ndarray]], n_bins: int,
                 l2_normalize: bool = True) -> sparse.csr_matrix:
    """Build one CSR matrix from a list of (bin_index, value) pairs.

    L2-normalising the rows here turns cosine similarity into a plain sparse
    matrix product later, which is the difference between a search that takes
    minutes and one that takes hours.
    """
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, (idx, _) in enumerate(rows):
        indptr[i + 1] = indptr[i] + idx.size
    indices = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, np.int32)
    data = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, np.float32)
    mat = sparse.csr_matrix((data.astype(np.float32), indices.astype(np.int32), indptr),
                            shape=(len(rows), n_bins))
    if l2_normalize:
        norms = np.sqrt(mat.multiply(mat).sum(axis=1)).A.ravel()
        norms[norms == 0] = 1.0
        mat = sparse.diags(1.0 / norms) @ mat
        mat = mat.tocsr()
    return mat
