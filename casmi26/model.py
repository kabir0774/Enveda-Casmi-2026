"""PeakFormer: a transformer that reads a spectrum and predicts a fingerprint.

Why a fingerprint and not SMILES directly: the published evidence is lopsided.
Fingerprint-then-decode reports ~28% top-1 on MassSpecGym; direct generative
models report 1-4%. The fingerprint is a bottleneck that forces the model to
commit to substructures, and substructures are what retrieval and decoding both
need.

Input encoding notes:

* m/z is encoded with sinusoidal features across many scales rather than fed as
  a raw number. A fragment at 121.0284 and one at 121.0648 are different
  molecules; a linear layer on a raw float cannot see that, while a bank of
  frequencies down to 0.01 Da can.
* Every peak also carries its neutral loss (precursor - m/z). Losses are what
  survive when a substituent shifts every fragment mass, so giving the model
  both views costs one extra embedding and buys analogue sensitivity.
* Collision energy and adduct are global tokens. The same molecule at 20 eV and
  80 eV produces different spectra, and a model that does not know the energy
  has to waste capacity inferring it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    dropout: float = 0.1
    fp_bits: int = 2048
    n_adducts: int = 16
    max_peaks: int = 128
    n_mz_freqs: int = 64
    min_wavelength: float = 0.01      # resolve 0.01 Da differences
    max_wavelength: float = 2000.0


class SinusoidalMass(nn.Module):
    """Multi-scale sinusoidal encoding of a mass in Daltons."""

    def __init__(self, n_freqs: int, min_wavelength: float, max_wavelength: float):
        super().__init__()
        steps = torch.arange(n_freqs, dtype=torch.float32) / max(n_freqs - 1, 1)
        wavelengths = min_wavelength * (max_wavelength / min_wavelength) ** steps
        self.register_buffer("inv_wavelengths", (2 * math.pi) / wavelengths)
        self.out_dim = 2 * n_freqs

    def forward(self, mass: torch.Tensor) -> torch.Tensor:
        # mass: (B, L) -> (B, L, 2*n_freqs)
        scaled = mass.unsqueeze(-1) * self.inv_wavelengths
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class PeakFormer(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.mz_enc = SinusoidalMass(cfg.n_mz_freqs, cfg.min_wavelength, cfg.max_wavelength)

        peak_in = self.mz_enc.out_dim * 2 + 2   # fragment m/z, neutral loss, intensity, log-intensity
        self.peak_proj = nn.Sequential(
            nn.Linear(peak_in, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model))

        self.adduct_emb = nn.Embedding(cfg.n_adducts, cfg.d_model)
        self.mode_emb = nn.Embedding(2, cfg.d_model)
        self.global_proj = nn.Sequential(
            nn.Linear(self.mz_enc.out_dim + 2, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        self.fp_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.fp_bits))
        # Auxiliary head: predict the neutral mass. Cheap, and it forces the
        # encoder to actually use the precursor rather than ignoring it.
        self.mass_head = nn.Linear(cfg.d_model, 1)

    def forward(self, mzs, intensities, mask, precursor_mz, adduct_id,
                mode_id, collision_energy):
        """
        mzs, intensities, mask: (B, L)   mask True = real peak
        precursor_mz, collision_energy:  (B,)
        adduct_id, mode_id:              (B,) long
        """
        losses = (precursor_mz.unsqueeze(1) - mzs).clamp(min=0.0)
        peak_feats = torch.cat([
            self.mz_enc(mzs),
            self.mz_enc(losses),
            intensities.unsqueeze(-1),
            torch.log1p(intensities.clamp(min=0)).unsqueeze(-1),
        ], dim=-1)
        x = self.peak_proj(peak_feats)

        g = self.global_proj(torch.cat([
            self.mz_enc(precursor_mz.unsqueeze(1)).squeeze(1),
            (collision_energy / 100.0).unsqueeze(-1),
            (precursor_mz / 1000.0).unsqueeze(-1),
        ], dim=-1))
        g = g + self.adduct_emb(adduct_id) + self.mode_emb(mode_id)

        B = x.shape[0]
        cls = self.cls.expand(B, -1, -1) + g.unsqueeze(1)
        x = torch.cat([cls, x], dim=1)
        pad_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1)

        h = self.encoder(x, src_key_padding_mask=~pad_mask)
        h = self.norm(h[:, 0])
        return {"fp_logits": self.fp_head(h),
                "mass": self.mass_head(h).squeeze(-1),
                "embedding": h}


def fingerprint_loss(fp_logits, fp_target, pos_weight: torch.Tensor | None = None):
    """BCE over bits.

    Fingerprints are sparse -- a few dozen on-bits out of 2048 -- so without
    pos_weight the model learns to predict all zeros and reports a great loss
    while being useless. Compute pos_weight once from the training set.
    """
    return F.binary_cross_entropy_with_logits(fp_logits, fp_target, pos_weight=pos_weight)


@torch.no_grad()
def bit_metrics(fp_logits, fp_target, threshold: float = 0.5) -> dict[str, float]:
    pred = (torch.sigmoid(fp_logits) >= threshold).float()
    tp = (pred * fp_target).sum()
    fp_ = (pred * (1 - fp_target)).sum()
    fn = ((1 - pred) * fp_target).sum()
    prec = (tp / (tp + fp_).clamp(min=1)).item()
    rec = (tp / (tp + fn).clamp(min=1)).item()
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return {"bit_precision": prec, "bit_recall": rec, "bit_f1": f1,
            "mean_on_bits_pred": pred.sum(dim=1).mean().item(),
            "mean_on_bits_true": fp_target.sum(dim=1).mean().item()}


@torch.no_grad()
def tanimoto_matrix(pred_probs: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
    """Continuous Tanimoto of predicted probabilities against a binary database.

    Thresholding the probabilities before comparing throws away the model's
    confidence; the published MIST+MolForge result thresholds for DECODING but
    retrieval works better on the soft values.
    """
    inter = pred_probs @ db.T
    denom = pred_probs.sum(dim=1, keepdim=True) + db.sum(dim=1).unsqueeze(0) - inter
    return inter / denom.clamp(min=1e-6)
