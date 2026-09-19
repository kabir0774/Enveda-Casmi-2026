"""Learning-rate range test -- find the usable LR band in one short run.

Instead of guessing a learning rate and burning a full training run to find
out, ramp the LR exponentially across a few hundred steps and watch where the
loss stops improving and starts diverging. Costs minutes, not hours.

  python scripts/lr_finder.py --train-parquet data/train.parquet --structures 8000

Read the output as: the LR where loss falls fastest is the aggressive end; the
LR where it turns upward is the ceiling. A common rule is to train at roughly
one order of magnitude below the divergence point, which this prints.
"""
from __future__ import annotations

import argparse, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from casmi26.model import ModelConfig, PeakFormer, fingerprint_loss
from casmi26.preprocess import CleanConfig
from casmi26.torch_data import DataConfig, SpectrumFingerprintDataset, collate, pos_weight_from


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--structures", type=int, default=8000)
    ap.add_argument("--max-spectra-per-structure", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--max-peaks", type=int, default=128)
    ap.add_argument("--pos-weight-cap", type=float, default=50.0)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--max-lr", type=float, default=3e-2)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_fingerprint import load_real, load_synthetic
    if a.synthetic or not a.train_parquet:
        spectra = load_synthetic(a.structures, a.max_spectra_per_structure, a.seed)
    else:
        spectra = load_real(a.train_parquet, a.structures,
                            a.max_spectra_per_structure, a.seed)
    ds = SpectrumFingerprintDataset(
        spectra, DataConfig(max_peaks=a.max_peaks, fp_bits=a.fp_bits,
                            clean=CleanConfig(max_peaks=a.max_peaks)))
    print(f"{len(ds)} spectra on {device}")

    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True, collate_fn=collate,
                    num_workers=a.workers, drop_last=True,
                    pin_memory=(device == "cuda"))
    pw = pos_weight_from(ds, cap=a.pos_weight_cap).to(device)

    cfg = ModelConfig(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads,
                      d_ff=a.d_model * 4, fp_bits=a.fp_bits, max_peaks=a.max_peaks)
    model = PeakFormer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.min_lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    gamma = (a.max_lr / a.min_lr) ** (1 / max(a.steps - 1, 1))
    lrs, losses, smooth, best = [], [], None, math.inf
    step = 0
    model.train()
    while step < a.steps:
        for batch in dl:
            if step >= a.steps:
                break
            lr = a.min_lr * (gamma ** step)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                out = model(batch["mzs"].to(device), batch["intensities"].to(device),
                            batch["mask"].to(device), batch["precursor_mz"].to(device),
                            batch["adduct_id"].to(device), batch["mode_id"].to(device),
                            batch["collision_energy"].to(device), batch["ce_known"].to(device))
                loss = fingerprint_loss(out["fp_logits"], batch["fp"].to(device), pw)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            l = loss.item()
            smooth = l if smooth is None else 0.9 * smooth + 0.1 * l
            lrs.append(lr)
            losses.append(smooth)
            best = min(best, smooth)
            step += 1
            if not math.isfinite(l) or smooth > 4 * best:
                print(f"diverged at step {step}, lr {lr:.2e}")
                step = a.steps
                break

    lrs, losses = np.array(lrs), np.array(losses)
    if lrs.size < 20:
        print("too few steps to judge -- lower --batch-size or raise --structures")
        return 1

    # steepest descent of loss with respect to log(lr)
    logs = np.log10(lrs)
    d = np.gradient(losses, logs)
    min_loss = int(np.argmin(losses))
    lo = max(int(0.1 * len(d)), 1)
    # Search for the steepest descent BEFORE the loss minimum. Past the minimum
    # the loss is climbing out of divergence, and the gradient there is noise.
    hi = max(min_loss, lo + 1)
    steep = int(np.argmin(d[lo:hi])) + lo

    print(f"\n{'lr':>12} {'smoothed loss':>15}")
    for i in range(0, len(lrs), max(len(lrs) // 20, 1)):
        mark = ""
        if i == steep:
            mark = "  <- steepest descent"
        if i == min_loss:
            mark += "  <- minimum"
        print(f"{lrs[i]:>12.2e} {losses[i]:>15.4f}{mark}")

    print(f"\nsteepest descent at lr = {lrs[steep]:.2e}")
    print(f"loss minimum      at lr = {lrs[min_loss]:.2e}")
    print(f"suggested training lr   = {lrs[min_loss] / 10:.2e}  "
          f"(order of magnitude below the minimum)")
    print(f"usable band roughly     = {lrs[steep]/3:.2e} to {lrs[min_loss]:.2e}")
    print("\nThis measures early-training behaviour only. A high LR that looks good "
          "here can still lose to a lower one over a full run -- confirm with the sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
