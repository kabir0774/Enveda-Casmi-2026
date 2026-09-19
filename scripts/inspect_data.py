"""First thing to run after downloading: what is actually in train.parquet.

  python scripts/inspect_data.py --train-parquet data/train.parquet
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from casmi26.data import LoadConfig, dataset_summary, scan_train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--test-parquet", default=None)
    a = ap.parse_args()

    raw = pl.scan_parquet(a.train_parquet)
    cols = raw.collect_schema().names()
    print("columns:", cols)
    print("rows:", raw.select(pl.len()).collect().item())
    for missing in ("molecule_id", "spectrum_id"):
        if missing not in cols:
            print(f"note: train has no '{missing}' -- synthesised by scan_train()")

    print("\nper source library (unfiltered):")
    print(dataset_summary(raw))

    lf = scan_train(a.train_parquet, LoadConfig())
    print(f"\nafter default filters: {lf.select(pl.len()).collect().item()} spectra, "
          f"{lf.select(pl.col('inchikey14').n_unique()).collect().item()} structures")

    print("\nadduct counts (filtered):")
    print(lf.group_by("adduct").agg(pl.len().alias("n")).sort("n", descending=True).collect())

    print("\nper library AFTER default filters (what you can actually train on):")
    print(lf.group_by("ingest_lib")
            .agg(pl.len().alias("spectra"),
                 pl.col("inchikey14").n_unique().alias("structures"))
            .sort("spectra", descending=True).collect())

    print("\nstructure overlap between the natural-product libraries and enveda-180:")
    np_keys = (lf.filter(pl.col("ingest_lib").is_in(["gnps", "riken", "enveda-np-examples"]))
                 .select("inchikey14").unique().collect()["inchikey14"])
    ev_keys = (lf.filter(pl.col("ingest_lib") == "enveda-180")
                 .select("inchikey14").unique().collect()["inchikey14"])
    inter = len(set(np_keys) & set(ev_keys))
    print(f"  natural-product structures: {len(np_keys)}")
    print(f"  enveda-180 structures:      {len(ev_keys)}")
    print(f"  shared:                     {inter}"
          f"  ({100*inter/max(len(np_keys),1):.1f}% of the NP set)")

    print("\npeak counts (filtered):")
    print(lf.select(pl.col("num_peaks")).collect()
            .describe().filter(pl.col("statistic").is_in(
                ["min", "25%", "50%", "75%", "max", "mean"])))

    print("\ncollision energy availability:")
    print(lf.select([
        pl.col("collision_energy_ev").is_null().sum().alias("null_ev"),
        pl.len().alias("total"),
    ]).collect())

    if a.test_parquet:
        t = pl.scan_parquet(a.test_parquet)
        print("\ntest columns:", t.collect_schema().names())
        print("test rows:", t.select(pl.len()).collect().item())
        print("test molecules:", t.select(pl.col("molecule_id").n_unique()).collect().item())
        print("\ntest adducts:")
        print(t.group_by("adduct").agg(pl.len().alias("n")).sort("n", descending=True).collect())
        print("\ntest spectra per molecule:")
        print(t.group_by("molecule_id").agg(pl.len().alias("n")).collect()
               .select("n").describe().filter(pl.col("statistic").is_in(
                   ["min", "50%", "max", "mean"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
