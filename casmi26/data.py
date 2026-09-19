"""Loading the competition parquet files.

Everything downstream consumes plain dicts, so this module is the only place
that knows the real column names. The point of that is portability: the
synthetic tests and the real run go through identical code paths.

Memory note: train.parquet is ~2.5M rows with two array columns. Read it lazily
and select only the columns a step needs -- materialising all 18 columns at once
is the fastest way to lose a machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

SHARED_COLUMNS = [
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
    "precursor_mz", "collision_energy_ev",
]
TRAIN_EXTRA = [
    "normalized_smiles", "inchikey14", "molecular_formula", "ingest_lib",
    "precursor_error_ppm", "num_peaks",
]

# The ten adducts that appear in the test set. Training carries many more;
# restricting to these is usually the right call for a model that will only
# ever be asked about these.
TEST_ADDUCTS = [
    "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
]

# The only two libraries acquired on the test instrument through the test
# pipeline. enveda-np-examples is also the only one with natural-product
# chemistry, which makes it the calibration set, not a training set.
INSTRUMENT_MATCHED = ["enveda-180", "enveda-np-examples"]


@dataclass(frozen=True)
class LoadConfig:
    """Filters applied at load time. All optional -- start permissive."""
    adducts: tuple[str, ...] | None = tuple(TEST_ADDUCTS)
    libraries: tuple[str, ...] | None = None
    min_peaks: int = 5
    max_precursor_error_ppm: float | None = 20.0
    max_precursor_mz: float = 1300.0
    require_collision_energy: bool = False


def scan_train(path: str | Path, cfg: LoadConfig = LoadConfig()) -> pl.LazyFrame:
    """Lazy, filtered view of train.parquet. Nothing is read until .collect()."""
    lf = pl.scan_parquet(str(path)).select(SHARED_COLUMNS + TRAIN_EXTRA)
    if cfg.adducts:
        lf = lf.filter(pl.col("adduct").is_in(list(cfg.adducts)))
    if cfg.libraries:
        lf = lf.filter(pl.col("ingest_lib").is_in(list(cfg.libraries)))
    if cfg.min_peaks:
        lf = lf.filter(pl.col("num_peaks") >= cfg.min_peaks)
    if cfg.max_precursor_error_ppm is not None:
        # precursor_error_ppm ships uncleaned, so it doubles as a label-quality
        # signal: a big ppm error means the labelled structure and the measured
        # precursor disagree, i.e. the label is probably wrong.
        lf = lf.filter(
            pl.col("precursor_error_ppm").is_null()
            | (pl.col("precursor_error_ppm").abs() <= cfg.max_precursor_error_ppm)
        )
    lf = lf.filter(pl.col("precursor_mz") <= cfg.max_precursor_mz)
    if cfg.require_collision_energy:
        lf = lf.filter(pl.col("collision_energy_ev").is_not_null())
    return lf


def scan_test(path: str | Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(path)).select(SHARED_COLUMNS)


def _first_or(value, default=0.0) -> float:
    """collision_energy_ev is a list: [20] is one acquisition, [20,40,60] a merge."""
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = [v for v in value if v is not None]
        return float(np.mean(arr)) if arr else float(default)
    return float(value)


def rows_to_spectra(df: pl.DataFrame, with_labels: bool = True) -> list[dict]:
    """Convert a materialised frame into the dict format the package uses."""
    out: list[dict] = []
    cols = df.columns
    for row in df.iter_rows(named=True):
        mz = np.asarray(row["ms2_mzs"], dtype=np.float64)
        it = np.asarray(row["ms2_normalized_intensities"], dtype=np.float64)
        if mz.size == 0 or mz.size != it.size:
            continue
        rec = {
            "molecule_id": row["molecule_id"],
            "spectrum_id": row["spectrum_id"],
            "mzs": mz,
            "intensities": it,
            "precursor_mz": float(row["precursor_mz"]),
            "adduct": row.get("adduct"),
            "ionization_mode": row.get("ionization_mode"),
            "collision_energy_ev": _first_or(row.get("collision_energy_ev"), 0.0),
            "base_peak_intensity": float(row.get("base_peak_intensity") or 0.0),
        }
        if with_labels and "normalized_smiles" in cols:
            rec["smiles"] = row["normalized_smiles"]
            rec["key"] = row["inchikey14"]
            rec["formula"] = row.get("molecular_formula")
            rec["ingest_lib"] = row.get("ingest_lib")
        out.append(rec)
    return out


def group_by_molecule(spectra: list[dict]) -> dict[str, list[dict]]:
    """Predictions are per molecule, so grouping is the first thing you do."""
    groups: dict[str, list[dict]] = {}
    for s in spectra:
        groups.setdefault(s["molecule_id"], []).append(s)
    return groups


def sample_structures(lf: pl.LazyFrame, n_structures: int, seed: int = 0,
                      max_spectra_per_structure: int = 8) -> pl.DataFrame:
    """Take a manageable subset: n distinct structures, capped spectra each.

    Use this for every experiment before the final run. Iterating on 20k
    structures and 100k spectra answers most questions in minutes, and almost
    nothing you learn at that scale changes at full scale.
    """
    keys = (lf.select("inchikey14").unique().collect()
            .sample(n=n_structures, seed=seed, shuffle=True)["inchikey14"])
    sub = lf.filter(pl.col("inchikey14").is_in(keys.to_list())).collect()
    if max_spectra_per_structure:
        sub = (sub.with_columns(pl.int_range(pl.len()).over("inchikey14").alias("_i"))
                  .filter(pl.col("_i") < max_spectra_per_structure)
                  .drop("_i"))
    return sub


def dataset_summary(lf: pl.LazyFrame) -> pl.DataFrame:
    """Spectra and distinct structures per source library. Run this first."""
    return (lf.group_by("ingest_lib")
              .agg(pl.len().alias("spectra"),
                   pl.col("inchikey14").n_unique().alias("structures"),
                   pl.col("precursor_mz").median().alias("median_precursor_mz"))
              .sort("spectra", descending=True)
              .collect())
