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
from casmi26.candidates import build_candidate_db, neutral_mass

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
    ap.add_argument("--extra-candidates", default=None,
                    help="directory holding bio_meta.pkl (+ bio_mass.npy) of an "
                         "external structure set such as ChEBI + LIPID MAPS. "
                         "Class 2 is by definition absent from the training "
                         "structures, so without this the database cannot "
                         "contain those answers at all")
    ap.add_argument("--mass-tol-ppm", type=float, default=10.0)
    ap.add_argument("--mass-tol-da", type=float, default=0.01)
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
    extra_keys = extra_smiles = extra_masses = None
    if a.extra_candidates:
        import pickle
        base = Path(a.extra_candidates)
        meta = pickle.load(open(base / "bio_meta.pkl", "rb"))
        extra_keys = np.asarray(meta["keys"])
        extra_smiles = np.asarray(meta["smiles"])
        mass_file = base / "bio_mass.npy"
        extra_masses = np.load(mass_file) if mass_file.exists() else None
        log(f"external candidates: {len(extra_keys)} structures from {base.name}")

    db = build_candidate_db(cat["inchikey14"].to_list(),
                            cat["normalized_smiles"].to_list(),
                            cat["molecular_formula"].to_list(),
                            extra_keys, extra_smiles, extra_masses)
    db_smiles = list(db.smiles)
    log(f"candidate database: {len(db)} structures searchable by mass")

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
        # The external set ships its own 6,930-bit fingerprints, which are not
        # the 2,048-bit Morgan the model predicts. Recomputing Morgan for the
        # whole database costs about 40s and guarantees both sides of the
        # similarity use the same representation.
        db_fp = np.stack([morgan_bits(s, cfg.fp_bits) for s in db_smiles])
        log(f"database fingerprints built ({db_fp.shape})")

    # ---- per molecule -----------------------------------------------------
    predictions: dict[str, list[str]] = {}
    failures = 0
    # Timed from the start of the loop, not from process start: the setup cost
    # is paid once regardless of how many molecules follow, so folding it into
    # a per-molecule rate overstates the remaining work by a wide margin.
    pool_sizes: list[int] = []
    t_loop = time.time()
    log(f"setup complete in {t_loop - T0:.0f}s; predicting {len(all_ids)} molecules")
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
                # The adduct says how much mass ionisation added, so the
                # neutral mass is recoverable and the pool shrinks from
                # ~340,000 to a few hundred before any similarity is computed.
                pool = None
                for q in qspecs:
                    m = neutral_mass(q["precursor_mz"], q.get("adduct"))
                    if m is not None:
                        pool = db.query_mass(m, a.mass_tol_da, a.mass_tol_ppm)
                        break
                if pool is None or pool.size == 0:
                    pool = np.arange(len(db_smiles))   # unknown adduct: no filter
                pool_sizes.append(int(pool.size))
                sims = tanimoto(pred, db_fp[pool])
                order = pool[np.argsort(sims)[::-1][:MAX_GUESSES]]
                ret_list = [db_smiles[j] for j in order]

            merged = list(dict.fromkeys(lib_list + ret_list))
            predictions[mol_id] = fill_slots(merged, filler, k=MAX_GUESSES)
        except Exception:
            failures += 1
            if failures <= 3:
                traceback.print_exc()
            predictions[mol_id] = fill_slots([], filler, k=MAX_GUESSES)

        if n % 50 == 0 or n == len(all_ids):
            rate = (time.time() - t_loop) / n
            log(f"{n}/{len(all_ids)} molecules  ({rate:.2f}s each, "
                f"~{rate * (len(all_ids) - n) / 60:.1f} min left)")

    if failures:
        log(f"WARNING: {failures} molecules fell back to the default list")

    # ---- write, validating first -----------------------------------------
    write_submission(predictions, a.out, expected_ids=all_ids)
    sizes = [len(v) for v in predictions.values()]
    log(f"wrote {a.out}: {len(predictions)} rows, "
        f"{min(sizes)}-{max(sizes)} guesses each")
    if pool_sizes:
        ps = np.array(pool_sizes)
        log(f"mass-filtered candidate pool: median {int(np.median(ps))}, "
            f"mean {ps.mean():.0f}, max {ps.max()} of {len(db_smiles)}")
    setup = t_loop - T0
    per_mol = (time.time() - t_loop) / max(len(all_ids), 1)
    log(f"budget: {setup:.0f}s setup + {per_mol:.2f}s/molecule "
        f"-> a full 400-molecule run is ~{(setup + 400 * per_mol) / 60:.0f} min "
        f"of the 540 min Kaggle allows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
