"""End-to-end test of the confidence-gate idea on synthetic data.

Question this answers: does gating retrieval on a calibrated confidence score
beat (a) library search alone and (b) the naive interleave that every public
notebook does?

Synthetic data cannot tell you the real score. It CAN tell you whether the
mechanism works, and whether the failure seen on the public leaderboard
reproduces.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from casmi26 import synth
from casmi26.fusion import fuse_hits, gate_features, rank_candidates
from casmi26.gate import (ConfidenceGate, DualGate, evaluate_dual,
                          evaluate_policy, oracle_merge, retrieval_features)
from casmi26.library import SpectralLibrary
from casmi26.metric import inchikey14, mrr_at_k, per_molecule_rr
from casmi26.splits import build_splits, group_kfold_by_key, report_by_class

RNG = np.random.default_rng(7)
N_MOL = 2976
MASS_TOL = 8.0              # candidate pool width, mimics a formula filter
PROPORTIONS = (0.40, 0.40, 0.20)


def main(fp_accuracy: float = 0.62, verbose: bool = True) -> dict:
    """fp_accuracy is the per-bit accuracy of the stand-in fingerprint head.

    It is the one knob that decides which regime you are in. Sweep it: the real
    competition is currently in the WEAK regime, where retrieval on its own is
    worse than library search.
    """
    FP_ACCURACY = fp_accuracy
    t0 = time.time()
    smiles = synth.enumerate_molecules(N_MOL, seed=1)
    keys = {s: inchikey14(s) for s in smiles}
    smiles = [s for s in smiles if keys[s]]
    # collapse any structures that share an InChIKey14
    by_key = {}
    for s in smiles:
        by_key.setdefault(keys[s], s)
    smiles = list(by_key.values())
    if verbose: print(f"{len(smiles)} distinct structures  ({time.time()-t0:.1f}s)")

    folds = group_kfold_by_key([keys[s] for s in smiles], n_folds=10, seed=3)
    train = [s for s in smiles if folds[keys[s]] >= 3]
    val = [s for s in smiles if folds[keys[s]] < 3]
    if verbose: print(f"train {len(train)}  val {len(val)}")

    splits = build_splits([keys[s] for s in val], PROPORTIONS, seed=11)
    n_by_class = {c: sum(1 for v in splits.values() if v.novelty_class == c) for c in (1, 2, 3)}
    if verbose: print("val class mix:", n_by_class)

    # ---- library: all train spectra, plus class-1 val molecules -------------
    lib_specs = []
    for i, s in enumerate(train):
        for sp in synth.make_spectra(s, 3, seed=i):
            lib_specs.append({**sp, "key": keys[s]})
    for i, s in enumerate(val):
        if splits[keys[s]].in_library:
            for sp in synth.make_spectra(s, 3, seed=50_000 + i):
                lib_specs.append({**sp, "key": keys[s]})
    if verbose: print(f"library spectra: {len(lib_specs)}")
    library = SpectralLibrary.build(lib_specs)

    # ---- candidate database: everything except class-3 val structures -------
    db_smiles = [s for s in smiles
                 if keys[s] not in splits or splits[keys[s]].in_database]
    db_fp = np.stack([synth.fingerprint(s) for s in db_smiles])
    db_mass = np.array([synth.exact_mass(s) for s in db_smiles])
    if verbose: print(f"candidate database: {len(db_smiles)} structures")

    # global fallback list: the most common structures, for padding to 25
    filler = db_smiles[:40]

    # ---- per-molecule inference -------------------------------------------
    molecules, answers = [], {}
    for i, s in enumerate(val):
        qspecs = synth.make_spectra(s, 3, seed=90_000 + i)
        hits = library.search(qspecs, top_k=120)
        cands = fuse_hits(hits, library)
        lib_list = rank_candidates(cands, k=25)

        meta = {
            "precursor_mz": qspecs[0]["precursor_mz"],
            "mean_n_peaks": float(np.mean([len(q["mzs"]) for q in qspecs])),
            "base_peak_intensity": qspecs[0]["base_peak_intensity"],
            "n_adducts": 1, "n_energies": len(qspecs),
        }
        feats = gate_features(cands, n_spectra=len(qspecs), query_meta=meta)

        pred_fp = synth.noisy_predicted_fingerprint(s, FP_ACCURACY, seed=i)
        mass_ok = np.flatnonzero(np.abs(db_mass - synth.exact_mass(s)) <= MASS_TOL)
        if mass_ok.size:
            sims = synth.tanimoto(pred_fp, db_fp[mass_ok])
            order = mass_ok[np.argsort(sims)[::-1][:25]]
            ret_list = [db_smiles[j] for j in order]
            ret_feats = retrieval_features(sims, int(mass_ok.size), meta["precursor_mz"])
        else:
            ret_list = []
            ret_feats = retrieval_features(np.zeros(0), 0, meta["precursor_mz"])

        molecules.append({"id": keys[s], "features": feats, "ret_features": ret_feats,
                          "library": lib_list, "retrieval": ret_list,
                          "filler": filler})
        answers[keys[s]] = s
    if verbose: print(f"inference done ({time.time()-t0:.1f}s)")

    # ---- split val in half: train the gate on one half, score on the other --
    half = group_kfold_by_key(list(answers), n_folds=2, seed=5)
    gate_train = [m for m in molecules if half[m["id"]] == 0]
    gate_test = [m for m in molecules if half[m["id"]] == 1]

    labels, ret_labels = [], []
    for m in gate_train:
        top = m["library"][0] if m["library"] else ""
        labels.append(int(bool(top) and inchikey14(top) == m["id"]))
        rtop = m["retrieval"][0] if m["retrieval"] else ""
        ret_labels.append(int(bool(rtop) and inchikey14(rtop) == m["id"]))
    if verbose:
        print(f"gate train: {len(gate_train)} molecules, "
              f"library top-1 correct {np.mean(labels):.3f}, "
              f"retrieval top-1 correct {np.mean(ret_labels):.3f}")

    gate = ConfidenceGate().fit([m["features"] for m in gate_train], labels)
    dual = DualGate().fit([m["features"] for m in gate_train], labels,
                          [m["ret_features"] for m in gate_train], ret_labels)

    test_answers = {m["id"]: answers[m["id"]] for m in gate_test}
    test_splits = {k: splits[k] for k in test_answers}

    if verbose:
        print("\npolicy                       MRR@25   class1   class2   class3")
        print("-" * 66)
    results = {}
    for name, kwargs in [
        ("library only (p=1)",      dict(fixed_p=1.0)),
        ("retrieval only (p=0)",    dict(fixed_p=0.0)),
        ("naive interleave (p=0.5)", dict(fixed_p=0.5)),
        ("GATED (learned p)",       dict(gate=gate)),
    ]:
        r = evaluate_policy(gate_test, kwargs.pop("gate", None), test_answers,
                            mrr_at_k, protect_threshold=0.6, **kwargs)
        rr = per_molecule_rr(r["predictions"], test_answers)
        rep = report_by_class(rr, test_splits)
        results[name] = rep
        if verbose:
            print(f"{name:27s} {rep['mrr_overall']:.4f}  "
                  f"{rep['mrr_class1']:.4f}  {rep['mrr_class2']:.4f}  {rep['mrr_class3']:.4f}")

    for label, pin in (("DUAL GATE (pinned)", True), ("DUAL GATE (soft blend)", False)):
        rd = evaluate_dual(gate_test, dual, test_answers, mrr_at_k, pin_winner=pin)
        rep = report_by_class(per_molecule_rr(rd["predictions"], test_answers), test_splits)
        results[label] = rep
        if verbose:
            print(f"{label:27s} {rep['mrr_overall']:.4f}  "
                  f"{rep['mrr_class1']:.4f}  {rep['mrr_class2']:.4f}  {rep['mrr_class3']:.4f}")

    ro = oracle_merge(gate_test, test_answers, mrr_at_k, inchikey14)
    rep = report_by_class(per_molecule_rr(ro["predictions"], test_answers), test_splits)
    results["oracle source choice"] = rep
    if verbose:
        print(f"{'oracle source choice':27s} {rep['mrr_overall']:.4f}  "
              f"{rep['mrr_class1']:.4f}  {rep['mrr_class2']:.4f}  {rep['mrr_class3']:.4f}")

    base = results["library only (p=1)"]["mrr_overall"]
    naive = results["naive interleave (p=0.5)"]["mrr_overall"]
    gated = results["GATED (learned p)"]["mrr_overall"]
    se = results["library only (p=1)"].get("se", 0.0)
    if verbose:
        print("-" * 66)
        print(f"n = {int(results['library only (p=1)']['n_molecules'])} molecules, SE ~ {se:.4f}")
        print(f"naive interleave vs library-only : {naive-base:+.4f}")
        print(f"gated            vs library-only : {gated-base:+.4f}")
        print(f"gated            vs naive        : {gated-naive:+.4f}")
        print(f"\ntotal {time.time()-t0:.1f}s")
    return {"results": results, "se": se,
            "n": int(results["library only (p=1)"]["n_molecules"])}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-accuracy", type=float, default=0.62)
    ap.add_argument("--sweep", action="store_true",
                    help="sweep retrieval strength instead of a single run")
    a = ap.parse_args()
    if not a.sweep:
        main(a.fp_accuracy)
    else:
        print(f"{'fp_acc':>7} {'lib-only':>9} {'retr-only':>10} {'naive':>8} "
              f"{'dual-pin':>9} {'dual-soft':>10} {'oracle':>8} {'soft-naive':>11}")
        print("-" * 84)
        for acc in (0.52, 0.56, 0.60, 0.65, 0.72, 0.80):
            out = main(acc, verbose=False)
            r = out["results"]
            lib = r["library only (p=1)"]["mrr_overall"]
            ret = r["retrieval only (p=0)"]["mrr_overall"]
            nai = r["naive interleave (p=0.5)"]["mrr_overall"]
            gat = r["GATED (learned p)"]["mrr_overall"]
            dua = r["DUAL GATE (soft blend)"]["mrr_overall"]
            dup = r["DUAL GATE (pinned)"]["mrr_overall"]
            orc = r["oracle source choice"]["mrr_overall"]
            best_single = max(lib, ret)
            print(f"{acc:>7.2f} {lib:>9.4f} {ret:>10.4f} {nai:>8.4f} {dup:>9.4f} "
                  f"{dua:>10.4f} {orc:>8.4f} {dua-nai:>+11.4f}")
        print(f"\nn = {out['n']} molecules per row, SE ~ {out['se']:.4f}")
