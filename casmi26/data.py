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

# The competition's data page says train.parquet carries "the test columns
# above, plus" the label columns. It does not. Verified against the real file
# (2,539,608 rows): train has NEITHER molecule_id NOR spectrum_id.
#
#   train (18): ingest_lib, normalized_smiles, inchikey, inchikey14,
#     molecular_formula, ionization_mode, instrument_type, adduct, adduct_orig,
#     precursor_mz, precursor_error_ppm, ms2_mzs, ms2_normalized_intensities,
#     num_peaks, base_peak_intensity, collision_energy_ev,
#     collision_energy_orig, collision_energy_orig_units
#   test (12): molecule_id, spectrum_id, + the acquisition columns
#
# So train gets a synthetic molecule_id = inchikey14, which is the right
# grouping anyway: the metric scores per structure, and two spectra of the same
# structure from different libraries belong to the same molecule.
TEST_COLUMNS = [
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
    "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
    "precursor_mz", "collision_energy_ev", "collision_energy_orig",
    "collision_energy_orig_units",
]
TRAIN_COLUMNS = [
    "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity", "adduct",
    "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_ev",
    "normalized_smiles", "inchikey", "inchikey14", "molecular_formula",
    "ingest_lib", "adduct_orig", "precursor_error_ppm", "num_peaks",
]

# Kept for callers that imported the old names.
SHARED_COLUMNS = TEST_COLUMNS
TRAIN_EXTRA = [c for c in TRAIN_COLUMNS if c not in TEST_COLUMNS]

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


def available(path: str | Path, wanted: list[str]) -> list[str]:
    """Intersect a wanted column list with what the file actually has.

    Defensive on purpose: the published schema and the shipped file disagree,
    and a re-run could change either.
    """
    have = set(pl.scan_parquet(str(path)).collect_schema().names())
    return [c for c in wanted if c in have]


def scan_train(path: str | Path, cfg: LoadConfig = LoadConfig()) -> pl.LazyFrame:
    """Lazy, filtered view of train.parquet. Nothing is read until .collect().

    Adds `spectrum_id` (row index) and `molecule_id` (= inchikey14), neither of
    which exists in the shipped file.
    """
    lf = (pl.scan_parquet(str(path))
            .select(available(path, TRAIN_COLUMNS))
            .with_row_index("spectrum_id")
            .with_columns(pl.col("inchikey14").alias("molecule_id"),
                          pl.col("spectrum_id").cast(pl.Utf8)))
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
    return pl.scan_parquet(str(path)).select(available(path, TEST_COLUMNS))


def _first_or(value, default=None):
    """collision_energy_ev is a list: [20] is one acquisition, [20,40,60] a merge.

    Returns None when the source recorded nothing. 14.8% of filtered training
    spectra are null here, and collapsing those to 0.0 would assert they were
    acquired at zero volts. The model gets a separate "known" flag instead.
    """
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = [v for v in value if v is not None]
        return float(np.mean(arr)) if arr else default
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
            "molecule_id": row.get("molecule_id") or row.get("inchikey14"),
            "spectrum_id": row.get("spectrum_id"),
            "mzs": mz,
            "intensities": it,
            "precursor_mz": float(row["precursor_mz"]),
            "adduct": row.get("adduct"),
            "ionization_mode": row.get("ionization_mode"),
            "collision_energy_ev": _first_or(row.get("collision_energy_ev")),
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


def structure_catalogue(path: str | Path, cfg: LoadConfig | None = None) -> pl.DataFrame:
    """Every distinct structure in train.parquet: key, SMILES, formula.

    Three columns and one row per structure, so this is cheap even though the
    file is 2.9 GB -- parquet only reads the columns asked for.

    This exists because the candidate database and the spectral library are
    different things and must be sized separately. A library is limited by how
    many spectra you can afford to search; a database is just a list of
    structures. Sampling 30k structures for both makes a molecular-formula pool
    about 3 candidates wide, which turns class-2 retrieval into a coin flip
    that looks like a triumph.
    """
    lf = pl.scan_parquet(str(path)).select(
        available(path, ["inchikey14", "normalized_smiles", "molecular_formula"]))
    if cfg is not None and cfg.max_precursor_error_ppm is not None:
        pass  # filtering on ppm would need that column; catalogue stays permissive
    return (lf.group_by("inchikey14")
              .agg(pl.col("normalized_smiles").first(),
                   pl.col("molecular_formula").first())
              .collect())
