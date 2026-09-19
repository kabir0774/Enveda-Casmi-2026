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

from casmi26.data import (LoadConfig, rows_to_spectra, sample_structures, scan_train,
                          structure_catalogue)
from casmi26.fusion import fuse_hits, gate_features, rank_candidates
from casmi26.gate import (DualGate, dual_merge, fill_slots, oracle_merge,
                          retrieval_features)
from casmi26.library import SpectralLibrary
from casmi26.metric import inchikey14, mrr_at_k, per_molecule_rr, validate_submission
from casmi26.splits import build_splits, group_kfold_by_key, report_by_class


def load_model(path: str, device: str):
    import torch
    from casmi26.model import ModelConfig, PeakFormer
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    model = PeakFormer(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"loaded {path} (epoch {ck.get('epoch')}, "
          f"val retrieval top1 {ck.get('metrics', {}).get('retrieval_top1', float('nan')):.4f})")
    return model, cfg, set(ck.get("train_keys") or [])


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
                    help="structures sampled to build the SPECTRAL LIBRARY")
    ap.add_argument("--full-database", action="store_true", default=True,
                    help="use every structure in train.parquet as the candidate "
                         "database, not just the sampled ones (default on)")
    ap.add_argument("--sampled-database", dest="full_database", action="store_false",
                    help="restrict the candidate database to the sampled "
                         "structures -- makes class-2 look far easier than it is")
    ap.add_argument("--val-molecules", type=int, default=600)
    ap.add_argument("--max-spectra-per-structure", type=int, default=6)
    ap.add_argument("--proportions", type=str, default="0.4,0.4,0.2",
                    help="assumed class 1,2,3 mix -- the real one is hidden")
    ap.add_argument("--pool-by", choices=["formula", "mass"], default="formula",
                    help="how the candidate pool is formed. formula = every "
                         "database structure with the same molecular formula "
                         "(realistic). mass = a precursor-mass window (gives an "
                         "unrealistically small pool)")
    ap.add_argument("--mass-tol", type=float, default=0.01,
                    help="Da window, used only with --pool-by mass")
    ap.add_argument("--query-spectra", type=int, default=3,
                    help="spectra per validation molecule used as the query; "
                         "these are always excluded from the library")
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

    # Structures the fingerprint model was trained on must not become
    # validation molecules: retrieval would then be scoring memorisation.
    trained_on: set[str] = set()
    if a.checkpoint:
        import torch
        _ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        trained_on = set(_ck.get("train_keys") or [])
        del _ck
        if not trained_on:
            print("WARNING: this checkpoint predates train_keys logging, so the "
                  "model may have been trained on some validation molecules. "
                  "Retrain to get an honest retrieval number.")

    eligible = [k for k in all_keys if k not in trained_on]
    print(f"{len(all_keys)} structures, {len(trained_on & set(all_keys))} of them "
          f"seen by the model -> {len(eligible)} eligible as validation molecules")
    rng = np.random.default_rng(a.seed)
    val_keys = list(rng.choice(eligible, size=min(a.val_molecules, len(eligible)),
                               replace=False))
    splits = build_splits(val_keys, props, seed=a.seed)
    mix = {c: sum(1 for v in splits.values() if v.novelty_class == c) for c in (1, 2, 3)}
    print(f"val molecules: {len(val_keys)}  class mix {mix}  (assumed {props})")

    # ---- query / library split -------------------------------------------
    # The query spectra of a validation molecule must NEVER be in the library.
    # Otherwise class-1 search finds a cosine-1.0 self-match and scores a
    # perfect 1.0000, which is a lookup of the identical measurement, not
    # library search. In the real competition the test spectra are fresh
    # acquisitions the library has never seen.
    #
    # Class 1 keeps its OTHER spectra in the library (that is what "reference
    # spectra exist" means). Classes 2 and 3 have all their spectra hidden.
    queries: dict[str, list[dict]] = {}
    keep_in_library: list[dict] = []
    demoted = 0
    for k in val_keys:
        specs = by_key[k]
        n_query = max(1, min(a.query_spectra, len(specs) - 1)) if len(specs) > 1 else 1
        queries[k] = specs[:n_query]
        rest = specs[n_query:]
        if splits[k].in_library:
            if rest:
                keep_in_library.extend(rest)
            else:
                # only one spectrum: no reference left, so it cannot be class 1
                splits[k].novelty_class = 2
                splits[k].in_library = False
                demoted += 1
    val_set = set(val_keys)
    lib_specs = [s for s in spectra if s["key"] not in val_set] + keep_in_library
    if demoted:
        print(f"demoted {demoted} class-1 molecules to class 2 "
              f"(only one spectrum, nothing left for the library)")
    mix = {c: sum(1 for v in splits.values() if v.novelty_class == c) for c in (1, 2, 3)}
    print(f"library: {len(lib_specs)} spectra "
          f"({len(keep_in_library)} of them other spectra of class-1 val molecules); "
          f"final class mix {mix}")
    library = SpectralLibrary.build(lib_specs)
    print(f"library built ({time.time()-t0:.0f}s)")

    # ---- candidate database: sized independently of the library -------------
    if a.full_database:
        cat = structure_catalogue(a.train_parquet)
        cat_keys = cat["inchikey14"].to_list()
        cat_smiles = cat["normalized_smiles"].to_list()
        cat_formula = [f or "" for f in cat["molecular_formula"].to_list()]
        # The file's inchikey14 and the metric's are not the same thing: the
        # metric applies RDKit tautomer canonicalisation first, so two entries
        # the file calls different can collapse to one key at scoring time. A
        # class-3 structure excluded by file key can therefore still be present
        # under a tautomer, which shows up as a non-zero class-3 score.
        drop_file_keys = {k for k, v in splits.items() if not v.in_database}
        drop_metric = {inchikey14(smiles_of[k]) for k in drop_file_keys
                       if k in smiles_of}
        drop_metric.discard(None)
        keep = [i for i, k in enumerate(cat_keys)
                if k not in drop_file_keys
                and inchikey14(cat_smiles[i]) not in drop_metric]
        db_keys = [cat_keys[i] for i in keep]
        db_smiles = [cat_smiles[i] for i in keep]
        db_formula = np.array([cat_formula[i] for i in keep])
        db_mass = np.zeros(len(db_keys))  # unused when pooling by formula
        print(f"candidate database: {len(db_keys)} structures "
              f"(full catalogue from train.parquet)")
    else:
        db_keys = [k for k in all_keys if k not in splits or splits[k].in_database]
        db_smiles = [smiles_of[k] for k in db_keys]
        db_mass = np.array([by_key[k][0]["precursor_mz"] for k in db_keys])
        db_formula = np.array([by_key[k][0].get("formula") or "" for k in db_keys])
        print(f"candidate database: {len(db_keys)} sampled structures "
              f"(class-2 numbers from this are optimistic)")

    model = cfg = None
    db_fp = None
    if a.checkpoint:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cfg, _ = load_model(a.checkpoint, device)
        from casmi26.torch_data import morgan_bits
        db_fp = np.stack([morgan_bits(s, cfg.fp_bits) for s in db_smiles])
        print(f"database fingerprints built ({time.time()-t0:.0f}s)")
    else:
        print("no checkpoint -> retrieval disabled, library search only")

    filler = db_smiles[:40]
    molecules, answers = [], {}
    pool_sizes = []
    class1_ranks: list[int] = []
    for n, key in enumerate(val_keys):
        qspecs = queries[key]
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
            # Candidate pool. Formula matching is the realistic filter -- it is
            # what SIRIUS-style pipelines do -- and it is what the real class-2
            # task faces. A narrow precursor-mass window instead produces a pool
            # of one or two candidates and a meaninglessly high class-2 score.
            if a.pool_by == "formula" and qspecs[0].get("formula"):
                pool = np.flatnonzero(db_formula == qspecs[0]["formula"])
            else:
                pool = np.flatnonzero(np.abs(db_mass - qspecs[0]["precursor_mz"]) <= a.mass_tol)
            pool_sizes.append(int(pool.size))
            if pool.size:
                sims = tanimoto(pred, db_fp[pool])
                order = pool[np.argsort(sims)[::-1][:25]]
                ret_list = [db_smiles[j] for j in order]
                ret_feats = retrieval_features(sims, int(pool.size), meta["precursor_mz"])

        if splits[key].novelty_class == 1:
            target = inchikey14(smiles_of[key])
            rank = next((i for i, s in enumerate(lib_list, 1)
                         if inchikey14(s) == target), None)
            class1_ranks.append(rank if rank else 0)

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
        results["best-source oracle"] = report_by_class(
            per_molecule_rr(oracle_merge(gtest, test_answers, mrr_at_k, inchikey14)["predictions"],
                            test_answers), test_splits)

    if class1_ranks:
        r = np.array(class1_ranks)
        found = r > 0
        print(f"\nclass-1 library search: correct structure in the top 25 for "
              f"{found.sum()}/{len(r)} molecules; of those, rank 1 for "
              f"{(r == 1).sum()}, median rank {int(np.median(r[found])) if found.any() else 0}")
        print("  If recall is high but rank-1 is low, the similarity SCORING is "
              "the problem, not the search. If recall itself is low, the library "
              "does not contain a usable reference.")

    if pool_sizes:
        ps = np.array(pool_sizes)
        print(f"\ncandidate pool per molecule ({a.pool_by}): median {int(np.median(ps))}, "
              f"mean {ps.mean():.0f}, max {ps.max()}, empty for {(ps == 0).sum()} molecules")
        print("  NOTE: the real class-2 task searches PubChem/COCONUT, where a "
              "formula pool is typically 10^2-10^4 candidates. A small pool here "
              "makes class-2 look far easier than it is.")

    print(f"\n{'policy':<20} {'MRR@25':>8} {'class1':>8} {'class2':>8} {'class3':>8}")
    print("-" * 56)
    for name, r in results.items():
        print(f"{name:<20} {r['mrr_overall']:>8.4f} {r['mrr_class1']:>8.4f} "
              f"{r['mrr_class2']:>8.4f} {r['mrr_class3']:>8.4f}")
    n = int(results["library only"]["n_molecules"])
    se = results["library only"].get("se", 0.0)
    print("-" * 56)
    print(f"n = {n} molecules, SE ~ {se:.4f}")
    print("best-source oracle picks which LIST goes first; it is not an upper "
          "bound over all merges, so a blend can legitimately beat it.")
    print(f"public leaderboard for reference: herd 0.339, top 0.362 (19 Sep 2026)")
    print(f"\ntotal {time.time()-t0:.0f}s")

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
