"""Hyperparameter search by random sampling over short proxy runs.

Grid search wastes budget: it spends equal effort on parameters that matter and
parameters that do not. Random search over the same budget covers each
individual parameter at far more distinct values, which is why it finds better
configurations for the same GPU hours.

  # 12 trials, each a short run, then a ranked table
  python scripts/sweep.py --train-parquet data/train.parquet --trials 12 \
      --structures 15000 --epochs 8 --out-root runs/sweep1

Each trial is a separate process, so a crash or an OOM costs one trial rather
than the sweep. Completed trials are skipped on re-run, so it is resumable.

Two warnings that matter more than the search itself:

  * Trials are ranked on val retrieval top-1, NOT val loss. Those two disagree
    in this problem -- val loss bottomed at epoch 10 in an earlier run while
    retrieval kept improving to epoch 33.
  * A short proxy run can rank configurations differently from a full run,
    especially for learning rate and model size. Take the top two or three and
    confirm them at full scale before believing the ordering.
"""
from __future__ import annotations

import argparse, json, itertools, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Search space. Log-uniform where the parameter spans orders of magnitude.
SPACE = {
    "lr":              ("loguniform", 1e-4, 1.5e-3),
    "batch_size":      ("choice", [256, 512, 1024]),
    "d_model":         ("choice", [256, 384]),
    "layers":          ("choice", [4, 6, 8]),
    "max_peaks":       ("choice", [128, 256]),
    "dropout":         ("choice", [0.05, 0.1, 0.2]),
    "weight_decay":    ("choice", [0.01, 0.05]),
    "pos_weight_cap":  ("choice", [50.0, 200.0]),
}


def sample(rng: np.random.Generator) -> dict:
    out = {}
    for k, spec in SPACE.items():
        if spec[0] == "loguniform":
            lo, hi = spec[1], spec[2]
            out[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            choices = spec[1]
            out[k] = choices[int(rng.integers(len(choices)))]
    # heads must divide d_model
    out["heads"] = 8
    return out


def run_trial(cfg: dict, args, out_dir: Path) -> dict | None:
    hist = out_dir / "history.json"
    if hist.exists():
        print(f"  (already done, reusing {hist})")
    else:
        cmd = [sys.executable, str(ROOT / "scripts" / "train_fingerprint.py"),
               "--epochs", str(args.epochs),
               "--structures", str(args.structures),
               "--max-spectra-per-structure", str(args.max_spectra_per_structure),
               "--workers", str(args.workers),
               "--seed", str(args.seed),
               "--out", str(out_dir),
               "--lr", f"{cfg['lr']:.6g}",
               "--batch-size", str(cfg["batch_size"]),
               "--d-model", str(cfg["d_model"]),
               "--layers", str(cfg["layers"]),
               "--heads", str(cfg["heads"]),
               "--max-peaks", str(cfg["max_peaks"]),
               "--dropout", str(cfg["dropout"]),
               "--weight-decay", str(cfg["weight_decay"]),
               "--pos-weight-cap", str(cfg["pos_weight_cap"])]
        if args.synthetic:
            cmd.append("--synthetic")
        else:
            cmd += ["--train-parquet", args.train_parquet]
        if args.libraries:
            cmd += ["--libraries", args.libraries]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
            print(f"  FAILED: {' | '.join(tail)}")
            return None
    try:
        h = json.loads(hist.read_text())
    except Exception:
        return None
    if not h:
        return None
    best = max(h, key=lambda e: e.get("retrieval_top1", 0.0))
    return {**cfg,
            "retrieval_top1": best.get("retrieval_top1", 0.0),
            "retrieval_top20": best.get("retrieval_top20", 0.0),
            "bit_f1": best.get("bit_f1", 0.0),
            "val_loss": best.get("val_loss", float("nan")),
            "best_epoch": best.get("epoch", -1),
            "secs": h[-1].get("secs", 0.0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--libraries", default=None)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--structures", type=int, default=15000)
    ap.add_argument("--max-spectra-per-structure", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="runs/sweep")
    ap.add_argument("--lr-only", action="store_true",
                    help="vary only the learning rate, everything else fixed -- "
                         "use this when you want a clean answer about one knob")
    a = ap.parse_args()

    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    if a.lr_only:
        lrs = [1e-4, 2e-4, 3e-4, 5e-4, 8e-4, 1.2e-3]
        configs = [{"lr": lr, "batch_size": 512, "d_model": 256, "layers": 6,
                    "heads": 8, "max_peaks": 128, "dropout": 0.1,
                    "weight_decay": 0.01, "pos_weight_cap": 50.0} for lr in lrs]
    else:
        configs = [sample(rng) for _ in range(a.trials)]

    # A proxy run with too few optimizer steps ranks configurations by noise:
    # every trial plateaus on the trivial solution and the learning rate cannot
    # show itself. Check before burning the budget.
    est_spectra = int(a.structures * min(a.max_spectra_per_structure, 3) * 0.85)
    worst_bs = max(c["batch_size"] for c in configs)
    est_steps = (est_spectra // worst_bs) * a.epochs
    print(f"~{est_spectra} train spectra, ~{est_steps} optimizer steps at the "
          f"largest batch size ({worst_bs})")
    if est_steps < 500:
        print(f"REFUSING: {est_steps} steps per trial is too few to distinguish "
              f"configurations.\n  Raise --structures or --epochs, or the sweep "
              f"will rank noise. Aim for >= 1000 steps.")
        return 1

    t0 = time.time()
    results = []
    for i, cfg in enumerate(configs):
        name = f"t{i:02d}_lr{cfg['lr']:.1e}_b{cfg['batch_size']}_d{cfg['d_model']}_l{cfg['layers']}"
        print(f"\n[{i+1}/{len(configs)}] {name}  ({time.time()-t0:.0f}s elapsed)")
        r = run_trial(cfg, a, root / name)
        if r:
            r["name"] = name
            results.append(r)
            print(f"  top1 {r['retrieval_top1']:.4f}  top20 {r['retrieval_top20']:.4f}  "
                  f"bitF1 {r['bit_f1']:.3f}  (best epoch {r['best_epoch']})")
        (root / "results.json").write_text(json.dumps(results, indent=2))

    if not results:
        print("no trials completed")
        return 1
    results.sort(key=lambda r: -r["retrieval_top1"])

    print(f"\n{'rank':>4} {'top1':>7} {'top20':>7} {'lr':>9} {'bs':>5} {'dm':>4} "
          f"{'L':>2} {'pk':>4} {'drop':>5} {'wd':>5} {'pwcap':>6}")
    print("-" * 72)
    for i, r in enumerate(results, 1):
        print(f"{i:>4} {r['retrieval_top1']:>7.4f} {r['retrieval_top20']:>7.4f} "
              f"{r['lr']:>9.2e} {r['batch_size']:>5} {r['d_model']:>4} {r['layers']:>2} "
              f"{r['max_peaks']:>4} {r['dropout']:>5} {r['weight_decay']:>5} "
              f"{r['pos_weight_cap']:>6}")

    best = results[0]
    print(f"\nbest: {best['name']}  top1 {best['retrieval_top1']:.4f}")
    print(f"total {time.time()-t0:.0f}s over {len(results)} trials")

    # Which knobs actually mattered: correlation of each parameter with the score.
    if len(results) >= 6:
        print("\nrank correlation of each parameter with top-1 "
              "(near 0 = this knob did not matter at this scale):")
        scores = np.array([r["retrieval_top1"] for r in results])
        sr = np.argsort(np.argsort(scores))
        for k in SPACE:
            vals = np.array([float(r[k]) for r in results])
            if len(set(vals.tolist())) < 2:
                continue
            vr = np.argsort(np.argsort(vals))
            c = np.corrcoef(sr, vr)[0, 1]
            print(f"  {k:<16} {c:+.2f}")
        print("With few trials these are noisy -- treat them as hints, not conclusions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
