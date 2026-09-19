"""Class-stratified validation -- the only honest signal in this competition.

Two failure modes this module exists to prevent:

1. Splitting by spectrum. The same structure appears in several source
   libraries, so a random spectrum split leaves copies of the answer in the
   library and every model looks brilliant. Split by InChIKey14, always.

2. Reporting one aggregate number. The test set mixes three novelty classes in
   a hidden ratio. A change that adds 0.02 to class 2 while costing 0.05 on
   class 1 shows up as a small aggregate gain on a class-1-heavy split and a
   loss on the real test set. Score each class separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MoleculeSplit:
    """One validation molecule and the world it is allowed to see."""
    molecule_key: str
    novelty_class: int          # 1, 2 or 3
    in_library: bool            # are its reference spectra visible?
    in_database: bool           # is its structure in the candidate database?


def assign_novelty_classes(keys: list[str], proportions=(0.4, 0.4, 0.2),
                           seed: int = 0) -> dict[str, int]:
    """Assign each validation molecule a simulated novelty class.

    The real mix is hidden. Default (40/40/20) is a guess -- rerun your
    evaluation across several mixes and check the ranking of your approaches is
    stable, rather than tuning to one assumed ratio.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(proportions, dtype=float)
    p = p / p.sum()
    draws = rng.choice([1, 2, 3], size=len(keys), p=p)
    return {k: int(c) for k, c in zip(keys, draws)}


def build_splits(val_keys: list[str], proportions=(0.4, 0.4, 0.2),
                 seed: int = 0) -> dict[str, MoleculeSplit]:
    classes = assign_novelty_classes(val_keys, proportions, seed)
    out = {}
    for k in val_keys:
        c = classes[k]
        out[k] = MoleculeSplit(
            molecule_key=k,
            novelty_class=c,
            in_library=(c == 1),       # class 1 alone keeps its reference spectra
            in_database=(c in (1, 2)), # class 3 is absent from PubChem/COCONUT too
        )
    return out


def library_row_mask(library_keys: np.ndarray,
                     splits: dict[str, MoleculeSplit]) -> np.ndarray:
    """Rows of the reference library a validation run is allowed to search.

    Every spectrum of a class-2 or class-3 molecule is hidden. Note this
    removes the molecule from the library for ALL validation molecules in the
    fold, which is slightly pessimistic but keeps one library per fold instead
    of one per molecule. Rebuild per molecule only if you can afford it.
    """
    hidden = {k for k, s in splits.items() if not s.in_library}
    return ~np.isin(library_keys, list(hidden)) if hidden else np.ones(len(library_keys), bool)


def database_mask(db_keys: np.ndarray, splits: dict[str, MoleculeSplit]) -> np.ndarray:
    """Entries of the candidate database a validation run is allowed to retrieve."""
    hidden = {k for k, s in splits.items() if not s.in_database}
    return ~np.isin(db_keys, list(hidden)) if hidden else np.ones(len(db_keys), bool)


def group_kfold_by_key(keys: list[str], n_folds: int = 5, seed: int = 0) -> dict[str, int]:
    """Assign each distinct structure to a fold. No structure spans folds."""
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    return {uniq[i]: int(pos % n_folds) for pos, i in enumerate(order)}


def report_by_class(rr_per_molecule: dict[str, float],
                    splits: dict[str, MoleculeSplit]) -> dict[str, float]:
    """Aggregate MRR overall and per novelty class."""
    out: dict[str, float] = {}
    vals = list(rr_per_molecule.values())
    out["mrr_overall"] = float(np.mean(vals)) if vals else 0.0
    out["n_molecules"] = float(len(vals))
    for c in (1, 2, 3):
        sel = [rr for k, rr in rr_per_molecule.items()
               if k in splits and splits[k].novelty_class == c]
        out[f"mrr_class{c}"] = float(np.mean(sel)) if sel else float("nan")
        out[f"n_class{c}"] = float(len(sel))
    # Standard error: the reason a 0.01 leaderboard move means nothing.
    if vals:
        out["se"] = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return out
