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
    print("columns:", raw.collect_schema().names())
    print("rows:", raw.select(pl.len()).collect().item())

    print("\nper source library (unfiltered):")
    print(dataset_summary(raw))

    lf = scan_train(a.train_parquet, LoadConfig())
    print(f"\nafter default filters: {lf.select(pl.len()).collect().item()} spectra, "
          f"{lf.select(pl.col('inchikey14').n_unique()).collect().item()} structures")

    print("\nadduct counts (filtered):")
    print(lf.group_by("adduct").agg(pl.len().alias("n")).sort("n", descending=True).collect())

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
