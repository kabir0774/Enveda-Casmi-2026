"""Train the fingerprint head. This is the part that needs the GPU.

  python scripts/train_fingerprint.py --train-parquet data/train.parquet \
      --structures 40000 --epochs 20 --batch-size 256 --out runs/fp_v1

Start with --synthetic to prove the loop works before touching real data.

Split is by InChIKey14, always. Validation reports bit-level F1 AND a retrieval
proxy (top-1/top-5/top-20 against a candidate database built from validation
structures) -- bit F1 alone is misleading, because a model can score well on
common bits and still never rank the right molecule first.
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from casmi26.model import PeakFormer, ModelConfig, bit_metrics, fingerprint_loss, tanimoto_matrix
from casmi26.torch_data import (DataConfig, SpectrumFingerprintDataset, collate,
                               morgan_bits, pos_weight_from)


def load_real(path: str, n_structures: int, max_spectra: int, seed: int):
    import polars as pl
    from casmi26.data import LoadConfig, rows_to_spectra, sample_structures, scan_train
    lf = scan_train(path, LoadConfig())
    df = sample_structures(lf, n_structures, seed=seed,
                           max_spectra_per_structure=max_spectra)
    return rows_to_spectra(df, with_labels=True)


def load_synthetic(n_structures: int, max_spectra: int, seed: int):
    from casmi26 import synth
    from casmi26.metric import inchikey14
    smiles = synth.enumerate_molecules(n_structures, seed=seed)
    out = []
    for i, s in enumerate(smiles):
        key = inchikey14(s)
        if not key:
            continue
        for sp in synth.make_spectra(s, min(max_spectra, 3), seed=i):
            out.append({**sp, "key": key, "molecule_id": key,
                        "adduct": "[M+H]+", "ionization_mode": "positive"})
    return out


def split_by_key(spectra: list[dict], val_frac: float = 0.15, seed: int = 0):
    keys = sorted({s["key"] for s in spectra})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])
    train = [s for s in spectra if s["key"] not in val_keys]
    val = [s for s in spectra if s["key"] in val_keys]
    return train, val


@torch.no_grad()
def retrieval_eval(model, loader, device, fp_bits: int, max_molecules: int = 2000):
    """Top-k retrieval against a database built from the validation structures.

    Predictions are pooled per molecule (mean of sigmoid over its spectra),
    which is the same aggregation inference will do.
    """
    model.eval()
    per_mol: dict[str, list[torch.Tensor]] = {}
    smiles_of: dict[str, str] = {}
    for batch in loader:
        out = model(batch["mzs"].to(device), batch["intensities"].to(device),
                    batch["mask"].to(device), batch["precursor_mz"].to(device),
                    batch["adduct_id"].to(device), batch["mode_id"].to(device),
                    batch["collision_energy"].to(device))
        probs = torch.sigmoid(out["fp_logits"]).cpu()
        for i, key in enumerate(batch["keys"]):
            per_mol.setdefault(key, []).append(probs[i])
    keys = list(per_mol)[:max_molecules]
    if len(keys) < 2:
        return {}
    pred = torch.stack([torch.stack(per_mol[k]).mean(0) for k in keys])
    db = torch.stack([torch.tensor(loader.dataset.key_to_fp[k]) for k in keys])
    sims = tanimoto_matrix(pred, db)
    order = sims.argsort(dim=1, descending=True)
    truth = torch.arange(len(keys)).unsqueeze(1)
    hit = (order == truth)
    ranks = hit.float().argmax(dim=1) + 1
    found = hit.any(dim=1)
    rr = torch.where(found, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float))
    return {
        "retrieval_top1": (ranks[found] == 1).float().mean().item() if found.any() else 0.0,
        "retrieval_top5": (ranks[found] <= 5).float().mean().item() if found.any() else 0.0,
        "retrieval_top20": (ranks[found] <= 20).float().mean().item() if found.any() else 0.0,
        "retrieval_mrr": rr.mean().item(),
        "retrieval_db_size": float(len(keys)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", type=str, default=None)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--structures", type=int, default=20000)
    ap.add_argument("--max-spectra-per-structure", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--max-peaks", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/fp")
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out={out_dir}")

    t0 = time.time()
    if args.synthetic or not args.train_parquet:
        spectra = load_synthetic(args.structures, args.max_spectra_per_structure, args.seed)
        print(f"SYNTHETIC data -- results say nothing about the real competition")
    else:
        spectra = load_real(args.train_parquet, args.structures,
                            args.max_spectra_per_structure, args.seed)
    print(f"{len(spectra)} spectra, {len({s['key'] for s in spectra})} structures "
          f"({time.time()-t0:.1f}s)")

    train_sp, val_sp = split_by_key(spectra, 0.15, args.seed)
    dcfg = DataConfig(max_peaks=args.max_peaks, fp_bits=args.fp_bits)
    train_ds = SpectrumFingerprintDataset(train_sp, dcfg)
    val_ds = SpectrumFingerprintDataset(val_sp, dcfg)
    # retrieval_eval needs the true fingerprint per structure
    for ds in (train_ds, val_ds):
        ds.key_to_fp = {s["key"]: morgan_bits(s["smiles"], args.fp_bits)
                        for s in ds.items}
    print(f"train {len(train_ds)} spectra / {len(train_ds.key_to_fp)} structures | "
          f"val {len(val_ds)} spectra / {len(val_ds.key_to_fp)} structures")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=args.workers,
                          drop_last=True, pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.workers,
                        pin_memory=(device == "cuda"))

    pw = pos_weight_from(train_ds).to(device)
    print(f"pos_weight: median {pw.median().item():.1f}  max {pw.max().item():.1f}")

    cfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                      d_ff=args.d_model * 4, fp_bits=args.fp_bits, max_peaks=args.max_peaks)
    model = PeakFormer(cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(train_dl) * args.epochs)
    warmup = int(steps * args.warmup_frac)

    def lr_at(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler(device, enabled=(args.amp and device == "cuda"))

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for batch in train_dl:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(args.amp and device == "cuda")):
                out = model(batch["mzs"].to(device), batch["intensities"].to(device),
                            batch["mask"].to(device), batch["precursor_mz"].to(device),
                            batch["adduct_id"].to(device), batch["mode_id"].to(device),
                            batch["collision_energy"].to(device))
                fp = batch["fp"].to(device)
                loss = fingerprint_loss(out["fp_logits"], fp, pw)
                loss = loss + 0.01 * torch.nn.functional.smooth_l1_loss(
                    out["mass"], batch["neutral_mass"].to(device) / 100.0)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item() * fp.shape[0]
            n += fp.shape[0]

        model.eval()
        vloss, vn, mets = 0.0, 0, []
        with torch.no_grad():
            for batch in val_dl:
                out = model(batch["mzs"].to(device), batch["intensities"].to(device),
                            batch["mask"].to(device), batch["precursor_mz"].to(device),
                            batch["adduct_id"].to(device), batch["mode_id"].to(device),
                            batch["collision_energy"].to(device))
                fp = batch["fp"].to(device)
                vloss += fingerprint_loss(out["fp_logits"], fp, pw).item() * fp.shape[0]
                vn += fp.shape[0]
                mets.append(bit_metrics(out["fp_logits"], fp))
        bit = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]} if mets else {}
        ret = retrieval_eval(model, val_dl, device, args.fp_bits)

        rec = {"epoch": epoch, "train_loss": running / max(n, 1),
               "val_loss": vloss / max(vn, 1), **bit, **ret,
               "lr": sched.get_last_lr()[0], "secs": time.time() - t0}
        history.append(rec)
        print(f"ep{epoch:3d} train {rec['train_loss']:.4f} val {rec['val_loss']:.4f} "
              f"bitF1 {rec.get('bit_f1',0):.3f} "
              f"top1 {rec.get('retrieval_top1',0):.3f} "
              f"top20 {rec.get('retrieval_top20',0):.3f} "
              f"({rec['secs']:.0f}s)")

        score = rec.get("retrieval_top1", 0.0)
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "config": cfg.__dict__,
                        "args": vars(args), "epoch": epoch, "metrics": rec},
                       out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    print(f"best val retrieval top-1: {best:.4f}  ->  {out_dir/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
