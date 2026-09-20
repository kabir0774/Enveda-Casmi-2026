"""Entropy similarity, for rescoring a shortlist.

The measured problem is specific: cosine search finds the correct structure 86%
of the time and puts it first only 55% of the time. Recall is fine; ordering is
not. That is exactly the shape of problem a two-stage design solves -- retrieve
cheaply, rerank expensively over a shortlist of ~150.

Entropy similarity (Li et al., Nature Methods 2021) is the published
replacement for cosine on MS/MS matching. Two properties matter here:

  * It weights peaks by how much information the spectrum carries. A spectrum
    dominated by one huge peak is low-entropy and its intensities get flattened
    before comparison, so a single shared base peak stops dominating the score.
  * It is computed on the MERGED spectrum, so a peak present in one spectrum
    and absent in the other is penalised directly rather than merely failing to
    contribute, which is what cosine does.

It cannot be written as a dot product, so it does not scale to a 500k-spectrum
library. Rescoring a shortlist is the point.
"""
from __future__ import annotations

import numpy as np

LN4 = float(np.log(4.0))


def _normalise(intensities: np.ndarray) -> np.ndarray:
    total = intensities.sum()
    if total <= 0:
        return intensities
    return intensities / total


def spectral_entropy(intensities: np.ndarray) -> float:
    """Shannon entropy of the intensity distribution."""
    p = _normalise(np.asarray(intensities, dtype=np.float64))
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def weight_intensities(intensities: np.ndarray, entropy_cutoff: float = 3.0) -> np.ndarray:
    """Li et al.'s entropy-dependent intensity transform.

    A low-entropy spectrum is one where a couple of peaks carry everything.
    Those are exactly the spectra where raw intensities mislead a similarity
    score, so their intensities are flattened by an exponent below 1. A
    high-entropy spectrum is left alone.
    """
    ints = np.asarray(intensities, dtype=np.float64)
    s = spectral_entropy(ints)
    if s >= entropy_cutoff:
        return _normalise(ints)
    w = 0.25 + 0.25 * s
    return _normalise(np.power(ints, w))


def _merge(mz_a, int_a, mz_b, int_b, tol: float):
    """Merge two peak lists, summing intensities of peaks within tol.

    Greedy nearest-match in m/z order: each peak pairs with at most one peak
    from the other spectrum, which is what stops a dense spectrum from matching
    everything in a sparse one.
    """
    i = j = 0
    merged = []
    na, nb = len(mz_a), len(mz_b)
    while i < na and j < nb:
        d = mz_a[i] - mz_b[j]
        if abs(d) <= tol:
            merged.append(int_a[i] + int_b[j])
            i += 1
            j += 1
        elif d < 0:
            merged.append(int_a[i])
            i += 1
        else:
            merged.append(int_b[j])
            j += 1
    merged.extend(int_a[i:])
    merged.extend(int_b[j:])
    return np.asarray(merged, dtype=np.float64)


def entropy_similarity(mz_a, int_a, mz_b, int_b, tol: float = 0.02,
                       entropy_cutoff: float = 3.0) -> float:
    """Entropy similarity in [0, 1]. 1 means identical."""
    mz_a = np.asarray(mz_a, dtype=np.float64)
    mz_b = np.asarray(mz_b, dtype=np.float64)
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0
    oa, ob = np.argsort(mz_a), np.argsort(mz_b)
    mz_a, mz_b = mz_a[oa], mz_b[ob]
    wa = weight_intensities(np.asarray(int_a, dtype=np.float64)[oa], entropy_cutoff)
    wb = weight_intensities(np.asarray(int_b, dtype=np.float64)[ob], entropy_cutoff)

    merged = _merge(mz_a, wa, mz_b, wb, tol)
    s_ab = spectral_entropy(merged)
    s_a = spectral_entropy(wa)
    s_b = spectral_entropy(wb)
    sim = 1.0 - (2.0 * s_ab - s_a - s_b) / LN4
    return float(np.clip(sim, 0.0, 1.0))


def neutral_loss_similarity(mz_a, int_a, prec_a, mz_b, int_b, prec_b,
                            tol: float = 0.02) -> float:
    """Entropy similarity computed on neutral losses instead of fragments.

    Survives a constant mass shift, so it matches analogues that differ by one
    substituent -- the case where fragment matching fails entirely.
    """
    la = float(prec_a) - np.asarray(mz_a, dtype=np.float64)
    lb = float(prec_b) - np.asarray(mz_b, dtype=np.float64)
    ka, kb = la >= 0, lb >= 0
    if ka.sum() == 0 or kb.sum() == 0:
        return 0.0
    return entropy_similarity(la[ka], np.asarray(int_a)[ka],
                              lb[kb], np.asarray(int_b)[kb], tol)
