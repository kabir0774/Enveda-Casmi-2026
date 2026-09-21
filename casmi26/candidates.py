"""The candidate database: what class-2 retrieval actually searches.

Two sources are merged:

  * every structure in train.parquet (~276k)
  * an external biological-chemistry set such as ChEBI + LIPID MAPS (~63k)

The external set is the point. Class 2 is defined as "the structure is in
PubChem or COCONUT but has no public reference spectrum", so a database built
only from training structures can never contain those answers -- which is why
class 2 measured 0.008 while everything else moved.

The other half of the job is the mass filter. The test file gives a precursor
m/z and an adduct, and the adduct says exactly how much mass the ionisation
added or removed. Subtracting it recovers the neutral mass, which cuts the
candidate pool from 340,000 to a few hundred before any similarity is computed.
Without it, retrieval ranks the correct structure against everything, which is
both slow and hopeless.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Monoisotopic mass added to the neutral molecule by each adduct in the test
# set. neutral_mass = precursor_mz - delta (charge is +/-1 throughout, so m/z
# equals the ion mass and no division is needed).
PROTON = 1.00727646
H = 1.0078250319
H2O = 18.0105646
NA = 22.98976928
K = 38.9637065
CL = 34.96885268
CH2O2 = 46.0054793      # formic acid
NH3 = 17.0265491

ADDUCT_DELTA: dict[str, float] = {
    "[M+H]+":        PROTON,
    "[M+NH4]+":      NH3 + PROTON,
    "[M-H2O+H]+":    PROTON - H2O,
    "[M-2H2O+H]+":   PROTON - 2 * H2O,
    "[M+Na]+":       NA - 0.00054858,        # sodium cation
    "[M+K]+":        K - 0.00054858,
    "[M-H]-":        -PROTON,
    "[M-H2O-H]-":    -PROTON - H2O,
    "[M+CH2O2-H]-":  CH2O2 - PROTON,
    "[M+Cl]-":       CL + 0.00054858,        # chloride anion
}

# Monoisotopic masses of the elements that appear in these formulas.
_ELEMENT_MASS = {
    "H": 1.0078250319, "D": 2.0141017778, "B": 11.0093055, "C": 12.0,
    "N": 14.0030740052, "O": 15.9949146221, "F": 18.0009380,
    "Na": 22.98976928, "Mg": 23.9850417, "Al": 26.98153853,
    "Si": 27.9769265327, "P": 30.97376151, "S": 31.97207069,
    "Cl": 34.96885271, "K": 38.9637069, "Ca": 39.9625912,
    "Fe": 55.9349421, "Co": 58.9332002, "Ni": 57.9353479,
    "Cu": 62.9296011, "Zn": 63.9291466, "As": 74.9215942,
    "Se": 79.9165196, "Br": 78.9183376, "I": 126.904473,
    "Hg": 201.970626, "Pt": 194.964774,
}
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula: str | None) -> float:
    """Monoisotopic mass from a molecular formula string.

    Parsing the formula beats calling RDKit on the SMILES: roughly 10 us
    against 150 us, which over 340,000 structures is 3 seconds instead of 50.
    Returns 0.0 for anything unparseable, and the caller filters those out.
    """
    if not formula:
        return 0.0
    total = 0.0
    pos = 0
    for m in _FORMULA_TOKEN.finditer(formula):
        if m.start() != pos:          # a character the pattern skipped
            return 0.0
        pos = m.end()
        el, n = m.group(1), m.group(2)
        mass = _ELEMENT_MASS.get(el)
        if mass is None:
            return 0.0
        total += mass * (int(n) if n else 1)
    return total if pos == len(formula) else 0.0


def neutral_mass(precursor_mz: float, adduct: str | None) -> float | None:
    """Neutral monoisotopic mass implied by a precursor m/z and its adduct."""
    if adduct is None:
        return None
    delta = ADDUCT_DELTA.get(adduct)
    if delta is None:
        return None
    m = float(precursor_mz) - delta
    return m if m > 0 else None


def consensus_neutral_mass(spectra: list[dict], spread: float = 0.5) -> float | None:
    """One neutral mass from all of a molecule's spectra.

    A molecule is measured as several adducts, and each gives an independent
    estimate of the same neutral mass. Taking the first resolvable one stakes
    the entire candidate window on a single adduct label being right; if it is
    wrong the window sits tens of Daltons away and the correct structure is
    never considered at all.

    The median across estimates survives one bad label, and estimates that
    disagree with it by more than `spread` are dropped before the final median.
    """
    masses = []
    for s in spectra:
        m = neutral_mass(s.get("precursor_mz", 0.0), s.get("adduct"))
        if m is not None:
            masses.append(m)
    if not masses:
        return None
    arr = np.asarray(masses, dtype=float)
    med = float(np.median(arr))
    agree = arr[np.abs(arr - med) <= spread]
    return float(np.median(agree)) if agree.size else med


@dataclass
class CandidateDB:
    """Structures searchable by neutral mass."""
    keys: np.ndarray
    smiles: np.ndarray
    masses: np.ndarray
    _order: np.ndarray | None = None
    _sorted_mass: np.ndarray | None = None

    def __post_init__(self):
        self._order = np.argsort(self.masses)
        self._sorted_mass = self.masses[self._order]

    def __len__(self) -> int:
        return len(self.keys)

    def query_mass(self, mass: float, tol_da: float = 0.01,
                   tol_ppm: float | None = 10.0) -> np.ndarray:
        """Indices of structures whose mass is within tolerance.

        A ppm tolerance is the physically meaningful one -- instrument accuracy
        scales with mass -- with an absolute floor so small molecules are not
        given an unreasonably tight window.
        """
        tol = tol_da
        if tol_ppm:
            tol = max(tol_da, mass * tol_ppm * 1e-6)
        lo = np.searchsorted(self._sorted_mass, mass - tol, side="left")
        hi = np.searchsorted(self._sorted_mass, mass + tol, side="right")
        return self._order[lo:hi]


def build_candidate_db(train_keys, train_smiles, train_formulas,
                       extra_keys=None, extra_smiles=None, extra_masses=None,
                       verbose: bool = True) -> CandidateDB:
    """Merge the training catalogue with an external structure set.

    Deduplicated on InChIKey14, training entries winning, since those carry a
    formula we can trust. Structures whose mass cannot be determined are
    dropped -- they could never be retrieved by a mass query anyway.
    """
    keys = list(train_keys)
    smiles = list(train_smiles)
    masses = [formula_mass(f) for f in train_formulas]

    n_train = len(keys)
    if extra_keys is not None:
        seen = set(keys)
        added = 0
        for i, k in enumerate(extra_keys):
            if k in seen:
                continue
            seen.add(k)
            keys.append(k)
            smiles.append(extra_smiles[i])
            masses.append(float(extra_masses[i]) if extra_masses is not None else 0.0)
            added += 1
        if verbose:
            print(f"  candidate db: {n_train} from train + {added} new external "
                  f"({len(extra_keys) - added} already present)")

    keys_a = np.array(keys)
    smiles_a = np.array(smiles)
    masses_a = np.array(masses, dtype=np.float64)
    ok = masses_a > 0
    if verbose and (~ok).sum():
        print(f"  dropped {int((~ok).sum())} structures with no usable mass")
    return CandidateDB(keys_a[ok], smiles_a[ok], masses_a[ok])
