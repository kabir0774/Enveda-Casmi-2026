"""Produce submission.csv from test.parquet. This is the file the Kaggle
notebook calls.

  python scripts/predict.py --test-parquet data/test.parquet \
      --train-parquet data/train.parquet \
      --checkpoint models/best.pt --out submission.csv

Design constraints this file exists to satisfy, all of them from the
competition rules rather than from taste:

  * The scoring notebook runs with INTERNET DISABLED. Nothing here downloads.
  * It gets at most 9 hours. Every stage prints elapsed time so a run that is
    going to overrun says so early rather than at hour eight.
  * A submission is rejected outright if one molecule_id is missing, repeated,
    null, or carries more than 25 guesses. So every molecule is emitted exactly
    once, with exactly 25 slots, and a per-molecule failure degrades to the
    fallback list instead of taking the run down.

A wrong guess costs nothing but the slot it occupies, so slots are always
filled to 25.
"""
from __future__ import annotations

import argparse, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl

from casmi26.data import (LoadConfig, rows_to_spectra, sample_structures,
                          scan_test, scan_train, structure_catalogue)
from casmi26.fusion import fuse_hits, rank_candidates
from casmi26.gate import fill_slots
from casmi26.library import SpectralLibrary
from casmi26.metric import MAX_GUESSES, write_submission
from casmi26.preprocess import BinConfig, CleanConfig, clean_peaks
from casmi26.rescore import rescore_candidates

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def load_model(path: str):
    import torch
    from casmi26.model import ModelConfig, PeakFormer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = PeakFormer(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    log(f"model loaded on {device} (epoch {ck.get('epoch')})")
    return model, cfg, device


def predict_fingerprints(model, spectra, device, fp_bits, batch=256):
    import torch
    from casmi26.torch_data import DataConfig, SpectrumFingerprintDataset, collate
    ds = SpectrumFingerprintDataset(spectra, DataConfig(fp_bits=fp_bits),
                                    with_targets=False)
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            b = collate([ds[j] for j in range(i, min(i + batch, len(ds)))])
            o = model(b["mzs"].to(device), b["intensities"].to(device),
                      b["mask"].to(device), b["precursor_mz"].to(device),
                      b["adduct_id"].to(device), b["mode_id"].to(device),
                      b["collision_energy"].to(device), b["ce_known"].to(device))
            out.append(torch.sigmoid(o["fp_logits"]).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, fp_bits), np.float32)


def tanimoto(pred: np.ndarray, db: np.ndarray) -> np.ndarray:
    inter = db @ pred
    denom = db.sum(axis=1) + pred.sum() - inter
    denom[denom <= 0] = 1.0
    return inter / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--library-structures", type=int, default=400000,
                    help="cap on structures in the spectral library; the "
                         "default exceeds the dataset so it means 'all'")
    ap.add_argument("--max-spectra-per-structure", type=int, default=6)
    ap.add_argument("--top-k-library", type=int, default=150)
    ap.add_argument("--loss-weight", type=float, default=0.5)
    ap.add_argument("--rescore", choices=["none", "entropy"], default="entropy")
    ap.add_argument("--rescore-top-n", type=int, default=100)
    ap.add_argument("--library-slots", type=int, default=15,
                    help="slots reserved for library candidates before "
                         "retrieval fills the rest")
    ap.add_argument("--limit", type=int, default=0,
                    help="only predict this many molecules (smoke testing)")
    a = ap.parse_args()

    # ---- test spectra -----------------------------------------------------
    test = scan_test(a.test_parquet).collect()
    # The list of molecules comes from the FILE, not from the parsed spectra.
    # rows_to_spectra drops a row whose peak arrays are empty or mismatched, so
    # a molecule whose every spectrum is unusable would silently vanish -- and
    # a missing molecule_id gets the whole submission rejected. Such a molecule
    # still gets its 25 fallback guesses.
    all_ids = sorted(test["molecule_id"].unique().to_list())
    test_spectra = rows_to_spectra(test, with_labels=False)
    groups: dict[str, list[dict]] = {m: [] for m in all_ids}
    for s in test_spectra:
        groups.setdefault(s["molecule_id"], []).append(s)
    empty = [m for m in all_ids if not groups[m]]
    log(f"test: {len(test_spectra)} spectra, {len(all_ids)} molecules")
    if empty:
        log(f"WARNING: {len(empty)} molecules have no usable spectra, "
            f"they get the fallback list: {empty[:5]}")
    if a.limit:
        all_ids = all_ids[: a.limit]
        log(f"LIMIT: predicting only {len(all_ids)} molecules (smoke test)")

    # ---- spectral library -------------------------------------------------
    lf = scan_train(a.train_parquet, LoadConfig())
    df = sample_structures(lf, a.library_structures, seed=0,
                           max_spectra_per_structure=a.max_spectra_per_structure)
    lib_specs = rows_to_spectra(df, with_labels=True)
    log(f"library: {len(lib_specs)} spectra / "
        f"{len({s['key'] for s in lib_specs})} structures")
    library = SpectralLibrary.build(lib_specs, bin_cfg=BinConfig(),
                                    store_peaks=(a.rescore != "none"))
    del lib_specs, df
    log("library built")

    # ---- candidate database and fallback ----------------------------------
    cat = structure_catalogue(a.train_parquet)
    db_smiles = cat["normalized_smiles"].to_list()
    db_formula = np.array([f or "" for f in cat["molecular_formula"].to_list()])
    log(f"candidate database: {len(db_smiles)} structures")

    # Fallback list, used to pad short lists and to rescue a molecule whose
    # prediction raises. Most-frequent structures in the training data are the
    # best blind guess available.
    common = (pl.scan_parquet(a.train_parquet)
                .group_by("normalized_smiles").agg(pl.len().alias("n"))
                .sort("n", descending=True).head(MAX_GUESSES).collect())
    filler = common["normalized_smiles"].to_list()
    log(f"fallback list: {len(filler)} structures")

    model = cfg = device = db_fp = None
    if a.checkpoint:
        model, cfg, device = load_model(a.checkpoint)
        from casmi26.torch_data import morgan_bits
        db_fp = np.stack([morgan_bits(s, cfg.fp_bits) for s in db_smiles])
        log("database fingerprints built")

    # ---- per molecule -----------------------------------------------------
    predictions: dict[str, list[str]] = {}
    failures = 0
    for n, mol_id in enumerate(all_ids, start=1):
        qspecs = groups[mol_id]
        if not qspecs:
            predictions[mol_id] = fill_slots([], filler, k=MAX_GUESSES)
            continue
        try:
            hits = library.search(qspecs, top_k=a.top_k_library,
                                  loss_weight=a.loss_weight)
            cands = fuse_hits(hits, library)
            if a.rescore == "entropy":
                for q in qspecs:
                    if "_mzs" not in q:
                        q["_mzs"], q["_ints"] = clean_peaks(
                            q["mzs"], q["intensities"], q["precursor_mz"], CleanConfig())
                cands = rescore_candidates(cands, qspecs, library,
                                           top_n=a.rescore_top_n,
                                           loss_weight=a.loss_weight)
            lib_list = rank_candidates(cands, k=a.library_slots)

            ret_list: list[str] = []
            if model is not None:
                pred = predict_fingerprints(model, qspecs, device, cfg.fp_bits).mean(axis=0)
                # No formula for the test molecules, so the pool is every
                # structure. That is the honest situation: without a formula
                # predictor this stage searches the whole catalogue.
                sims = tanimoto(pred, db_fp)
                order = np.argsort(sims)[::-1][:MAX_GUESSES]
                ret_list = [db_smiles[j] for j in order]

            merged = list(dict.fromkeys(lib_list + ret_list))
            predictions[mol_id] = fill_slots(merged, filler, k=MAX_GUESSES)
        except Exception:
            failures += 1
            if failures <= 3:
                traceback.print_exc()
            predictions[mol_id] = fill_slots([], filler, k=MAX_GUESSES)

        if n % 50 == 0 or n == len(all_ids):
            rate = (time.time() - T0) / n
            log(f"{n}/{len(all_ids)} molecules  ({rate:.2f}s each, "
                f"~{rate * (len(all_ids) - n) / 60:.1f} min left)")

    if failures:
        log(f"WARNING: {failures} molecules fell back to the default list")

    # ---- write, validating first -----------------------------------------
    write_submission(predictions, a.out, expected_ids=all_ids)
    sizes = [len(v) for v in predictions.values()]
    log(f"wrote {a.out}: {len(predictions)} rows, "
        f"{min(sizes)}-{max(sizes)} guesses each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
