"""Synthetic spectra for testing the pipeline.

This is NOT a chemistry simulator and nothing it produces says anything about
real fragmentation. Its only job is to give the pipeline data with the
structural properties that matter for testing:

  * the same molecule measured twice gives similar spectra
  * molecules sharing substructures give partially overlapping peaks
  * unrelated molecules do not
  * collision energy changes which peaks appear

Peaks are derived from Morgan fingerprint bits, so substructure sharing turns
directly into peak sharing. Use it to prove the code runs and the metric and
gate behave; never to estimate a real score.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

PROTON = 1.007276

_TEMPLATES = [
    "{a}c1ccc({b})cc1",
    "{a}c1ccc({b})nc1",
    "{a}C1CCC({b})CC1",
    "{a}c1cc({b})c2ccccc2c1",
    "{a}C1CCC({b})O1",
    "{a}c1ccc({b})s1",
]
_SUBS = ["C", "CC", "CCC", "CCCC", "O", "OC", "OCC", "OCCC", "N", "NC", "NCC",
         "C(=O)O", "C(=O)N", "C(=O)C", "Cl", "F", "Br", "CO", "CCO", "CCN",
         "C(C)C", "CC(C)C", "CN", "CS", "C#N", "C(F)(F)F"]

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def enumerate_molecules(limit: int = 1200, seed: int = 0) -> list[str]:
    """Distinct, valid, canonical SMILES built from templates and substituents."""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    combos = [(t, a, b) for t in _TEMPLATES for a in _SUBS for b in _SUBS]
    rng.shuffle(combos)
    for t, a, b in combos:
        smi = t.format(a=a, b=b)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        seen.add(canon)
        out.append(canon)
        if len(out) >= limit:
            break
    return out


def fingerprint(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(2048, dtype=np.float32)
    return np.asarray(_MORGAN.GetFingerprintAsNumPy(mol), dtype=np.float32)


def exact_mass(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    return float(Descriptors.ExactMolWt(mol)) if mol is not None else 0.0


def make_spectra(smiles: str, n_spectra: int = 3, seed: int = 0,
                 noise_peaks: int = 6, dropout: float = 0.35) -> list[dict]:
    """Several spectra of one molecule at different collision energies."""
    rng = np.random.default_rng(seed)
    fp = fingerprint(smiles)
    bits = np.flatnonzero(fp)
    mass = exact_mass(smiles)
    precursor = mass + PROTON

    # Each on-bit becomes a candidate fragment, deterministic per bit.
    base_mz = 40.0 + (bits % 1000) * 0.9 + (bits % 7) * 0.013
    base_mz = np.clip(base_mz, 40.0, max(precursor - 1.0, 45.0))
    base_int = 0.2 + ((bits * 2654435761) % 1000) / 1000.0

    out = []
    energies = np.linspace(20.0, 80.0, n_spectra)
    for i, ev in enumerate(energies):
        r = np.random.default_rng(seed * 1000 + i)
        # higher energy: favour low-mass fragments, drop the precursor
        weight = np.exp(-(base_mz / precursor) * (ev / 30.0))
        inten = base_int * weight
        keep = r.random(bits.size) > dropout
        mz, it = base_mz[keep], inten[keep]
        if mz.size == 0:
            mz, it = base_mz[:1], base_int[:1]
        # precursor survives mostly at low energy
        if r.random() < max(0.05, 1.0 - ev / 100.0):
            mz = np.append(mz, precursor)
            it = np.append(it, 0.6 * (1.0 - ev / 120.0))
        # instrument noise
        if noise_peaks:
            nz = r.uniform(40.0, precursor, size=noise_peaks)
            mz = np.concatenate([mz, nz])
            it = np.concatenate([it, r.uniform(0.0, 0.05, size=noise_peaks)])
        mz = mz + r.normal(0.0, 0.004, size=mz.size)   # mass accuracy jitter
        it = np.clip(it * r.uniform(0.9, 1.1, size=it.size), 1e-6, None)
        it = it / it.max()
        order = np.argsort(mz)
        out.append({
            "mzs": mz[order], "intensities": it[order],
            "precursor_mz": precursor, "collision_energy_ev": float(ev),
            "smiles": smiles, "base_peak_intensity": float(r.uniform(2e3, 5e5)),
        })
    return out


def noisy_predicted_fingerprint(smiles: str, accuracy: float = 0.75,
                                seed: int = 0) -> np.ndarray:
    """Stand-in for a trained fingerprint head.

    `accuracy` is the per-bit chance of getting a bit right. A real MIST-style
    encoder is better than random but far from perfect, and the retrieval
    quality it produces is what decides whether the gate has anything worth
    gating to.
    """
    rng = np.random.default_rng(seed)
    true = fingerprint(smiles)
    flip = rng.random(true.size) > accuracy
    pred = np.where(flip, 1.0 - true, true)
    return pred.astype(np.float32)


def tanimoto(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto of one vector against a matrix of vectors."""
    inter = B @ a
    denom = B.sum(axis=1) + a.sum() - inter
    denom[denom == 0] = 1.0
    return inter / denom
