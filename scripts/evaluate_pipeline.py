"""The only number comparable to the Kaggle leaderboard: MRR@25.

Everything the training script prints (bit F1, retrieval top-k) measures ONE
COMPONENT against a toy database. This script runs the whole pipeline the way a
submission does -- library search, candidate retrieval, fusion, 25 ranked SMILES
per molecule -- on held-out structures with simulated novelty classes, and
scores it with the competition metric.

  # library search only: this is the number to compare against the public 0.339
  python scripts/evaluate_pipeline.py --train-parquet data/train.parquet

  # with a trained fingerprint model driving class-2 retrieval
  python scripts/evaluate_pipeline.py --train-parquet data/train.parquet \
      --checkpoint runs/fp_all_40k/best.pt

Caveats that matter when you read the output:
  * The novelty-class mix is a guess (the real one is hidden). Sweep it.
  * The candidate database here is built from training structures, not PubChem.
    Real class-2 retrieval searches a far larger pool, so these class-2 numbers
    are optimistic.
  * Validation structures are held out by InChIKey14, so class-1 numbers are
    honest: the library genuinely contains no copy of a class-2 or class-3
    answer.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from casmi26.data import LoadConfig, rows_to_spectra, sample_structures, scan_train
from casmi26.fusion import fuse_hits, gate_features, rank_candidates
from casmi26.gate import (DualGate, dual_merge, fill_slots, oracle_merge,
                          retrieval_features)
from casmi26.library import SpectralLibrary
from casmi26.metric import inchikey14, mrr_at_k, per_molecule_rr, validate_submission
from casmi26.splits import build_splits, group_kfold_by_key, report_by_class


def load_model(path: str, device: str):
    import torch
    from casmi26.model import ModelConfig, PeakFormer
    ck = torch.load(path, map_location=device)
    cfg = ModelConfig(**ck["config"])
    model = PeakFormer(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {path} (epoch {ck.get('epoch')}, "
          f"val retrieval top1 {ck.get('metrics', {}).get('retrieval_top1', float('nan')):.4f})")
    return model, cfg


def predict_fingerprints(model, spectra: list[dict], device: str, fp_bits: int,
                         batch: int = 256) -> np.ndarray:
    """Mean of sigmoid over a molecule's spectra -- the inference-time pooling."""
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
    return np.concatenate(out) if out else np.zeros((0, fp_bits), dtype=np.float32)


def tanimoto(pred: np.ndarray, db: np.ndarray) -> np.ndarray:
    inter = db @ pred
    denom = db.sum(axis=1) + pred.sum() - inter
    denom[denom <= 0] = 1.0
    return inter / denom


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="fingerprint model; omitted = library search only")
    ap.add_argument("--structures", type=int, default=30000,
                    help="structures sampled to build library + database")
    ap.add_argument("--val-molecules", type=int, default=600)
    ap.add_argument("--max-spectra-per-structure", type=int, default=6)
    ap.add_argument("--proportions", type=str, default="0.4,0.4,0.2",
                    help="assumed class 1,2,3 mix -- the real one is hidden")
    ap.add_argument("--mass-tol", type=float, default=0.01,
                    help="Da window for the candidate pool (stands in for a "
                         "molecular-formula filter)")
    ap.add_argument("--top-k-library", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    props = tuple(float(x) for x in a.proportions.split(","))

    lf = scan_train(a.train_parquet, LoadConfig())
    df = sample_structures(lf, a.structures, seed=a.seed,
                           max_spectra_per_structure=a.max_spectra_per_structure)
    spectra = rows_to_spectra(df, with_labels=True)
    print(f"{len(spectra)} spectra / {len({s['key'] for s in spectra})} structures "
          f"({time.time()-t0:.0f}s)")

    by_key: dict[str, list[dict]] = {}
    for s in spectra:
        by_key.setdefault(s["key"], []).append(s)
    all_keys = sorted(by_key)
    smiles_of = {k: by_key[k][0]["smiles"] for k in all_keys}

    rng = np.random.default_rng(a.seed)
    val_keys = list(rng.choice(all_keys, size=min(a.val_molecules, len(all_keys)),
                               replace=False))
    splits = build_splits(val_keys, props, seed=a.seed)
    mix = {c: sum(1 for v in splits.values() if v.novelty_class == c) for c in (1, 2, 3)}
    print(f"val molecules: {len(val_keys)}  class mix {mix}  (assumed {props})")

    # ---- library: every structure except the class-2 and class-3 val ones ----
    hidden = {k for k, v in splits.items() if not v.in_library}
    lib_specs = [s for s in spectra if s["key"] not in hidden]
    print(f"library: {len(lib_specs)} spectra  (hiding {len(hidden)} val structures)")
    library = SpectralLibrary.build(lib_specs)
    print(f"library built ({time.time()-t0:.0f}s)")

    # ---- candidate database: everything except class-3 val structures --------
    db_keys = [k for k in all_keys if k not in splits or splits[k].in_database]
    db_smiles = [smiles_of[k] for k in db_keys]
    db_mass = np.array([by_key[k][0]["precursor_mz"] for k in db_keys])

    model = cfg = None
    db_fp = None
    if a.checkpoint:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg = load_model(a.checkpoint, device)
        from casmi26.torch_data import morgan_bits
        db_fp = np.stack([morgan_bits(s, cfg.fp_bits) for s in db_smiles])
        print(f"database: {len(db_keys)} structures, fingerprints built "
              f"({time.time()-t0:.0f}s)")
    else:
        print(f"database: {len(db_keys)} structures (no model -> retrieval disabled)")

    filler = db_smiles[:40]
    molecules, answers = [], {}
    for n, key in enumerate(val_keys):
        qspecs = by_key[key]
        hits = library.search(qspecs, top_k=a.top_k_library)
        cands = fuse_hits(hits, library)
        lib_list = rank_candidates(cands, k=25)

        meta = {"precursor_mz": qspecs[0]["precursor_mz"],
                "mean_n_peaks": float(np.mean([len(q["mzs"]) for q in qspecs])),
                "base_peak_intensity": qspecs[0].get("base_peak_intensity", 0.0),
                "n_adducts": len({q.get("adduct") for q in qspecs}),
                "n_energies": len({q.get("collision_energy_ev") for q in qspecs})}
        feats = gate_features(cands, n_spectra=len(qspecs), query_meta=meta)

        ret_list, ret_feats = [], retrieval_features(np.zeros(0), 0, meta["precursor_mz"])
        if model is not None:
            pred = predict_fingerprints(model, qspecs, device, cfg.fp_bits).mean(axis=0)
            pool = np.flatnonzero(np.abs(db_mass - qspecs[0]["precursor_mz"]) <= a.mass_tol)
            if pool.size:
                sims = tanimoto(pred, db_fp[pool])
                order = pool[np.argsort(sims)[::-1][:25]]
                ret_list = [db_smiles[j] for j in order]
                ret_feats = retrieval_features(sims, int(pool.size), meta["precursor_mz"])

        molecules.append({"id": key, "features": feats, "ret_features": ret_feats,
                          "library": lib_list, "retrieval": ret_list, "filler": filler})
        answers[key] = smiles_of[key]
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(val_keys)} molecules ({time.time()-t0:.0f}s)")

    # ---- gate trained on half, scored on the other half ---------------------
    half = group_kfold_by_key(list(answers), n_folds=2, seed=a.seed + 1)
    gtrain = [m for m in molecules if half[m["id"]] == 0]
    gtest = [m for m in molecules if half[m["id"]] == 1]
    lib_lab = [int(bool(m["library"]) and inchikey14(m["library"][0]) == m["id"]) for m in gtrain]
    ret_lab = [int(bool(m["retrieval"]) and inchikey14(m["retrieval"][0]) == m["id"]) for m in gtrain]
    gate = DualGate().fit([m["features"] for m in gtrain], lib_lab,
                          [m["ret_features"] for m in gtrain], ret_lab)

    test_answers = {m["id"]: answers[m["id"]] for m in gtest}
    test_splits = {k: splits[k] for k in test_answers}

    def score(preds: dict[str, list[str]]) -> dict:
        validate_submission(preds, test_answers.keys())
        return report_by_class(per_molecule_rr(preds, test_answers), test_splits)

    results: dict[str, dict] = {}
    results["library only"] = score({
        m["id"]: fill_slots(m["library"], m["filler"]) for m in gtest})
    if model is not None:
        results["retrieval only"] = score({
            m["id"]: fill_slots(m["retrieval"], m["filler"]) for m in gtest})
        results["naive 50/50"] = score({
            m["id"]: fill_slots(dual_merge(m["library"], m["retrieval"], 0.5, 0.5), m["filler"])
            for m in gtest})
        p_lib, p_ret = gate.predict([m["features"] for m in gtest],
                                    [m["ret_features"] for m in gtest])
        results["GATED"] = score({
            m["id"]: fill_slots(dual_merge(m["library"], m["retrieval"], float(pl), float(pr),
                                           pin_winner=False), m["filler"])
            for m, pl, pr in zip(gtest, p_lib, p_ret)})
        results["oracle source"] = report_by_class(
            per_molecule_rr(oracle_merge(gtest, test_answers, mrr_at_k, inchikey14)["predictions"],
                            test_answers), test_splits)

    print(f"\n{'policy':<18} {'MRR@25':>8} {'class1':>8} {'class2':>8} {'class3':>8}")
    print("-" * 54)
    for name, r in results.items():
        print(f"{name:<18} {r['mrr_overall']:>8.4f} {r['mrr_class1']:>8.4f} "
              f"{r['mrr_class2']:>8.4f} {r['mrr_class3']:>8.4f}")
    n = int(results["library only"]["n_molecules"])
    se = results["library only"].get("se", 0.0)
    print("-" * 54)
    print(f"n = {n} molecules, SE ~ {se:.4f}")
    print(f"public leaderboard for reference: herd 0.339, top 0.362 (19 Sep 2026)")
    print(f"\ntotal {time.time()-t0:.0f}s")

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
