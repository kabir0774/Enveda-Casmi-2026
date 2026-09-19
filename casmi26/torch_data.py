"""Dataset and collation for fingerprint training."""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import Dataset

from .data import TEST_ADDUCTS
from .preprocess import CleanConfig, clean_peaks

RDLogger.DisableLog("rdApp.*")

ADDUCT_TO_ID = {a: i for i, a in enumerate(TEST_ADDUCTS)}
UNKNOWN_ADDUCT = len(TEST_ADDUCTS)


@functools.lru_cache(maxsize=1)
def _morgan(fp_bits: int, radius: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_bits)


@functools.lru_cache(maxsize=500_000)
def morgan_bits(smiles: str, fp_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.asarray(_morgan(fp_bits, radius).GetFingerprintAsNumPy(mol), dtype=np.float32)


@dataclass
class DataConfig:
    max_peaks: int = 128
    fp_bits: int = 2048
    fp_radius: int = 2
    clean: CleanConfig = CleanConfig()


class SpectrumFingerprintDataset(Dataset):
    """One spectrum -> one fingerprint target.

    Training is per spectrum even though scoring is per molecule. That is
    deliberate: it multiplies the training signal by the number of spectra per
    structure, and the per-molecule aggregation happens at inference. Just make
    sure the train/val split is by structure, or the same molecule's other
    spectra leak the answer.
    """

    def __init__(self, spectra: list[dict], cfg: DataConfig = DataConfig(),
                 with_targets: bool = True, precompute: bool = True,
                 verbose: bool = False):
        self.cfg = cfg
        self.with_targets = with_targets
        self.items = []
        for s in spectra:
            if with_targets:
                fp = morgan_bits(s.get("smiles", ""), cfg.fp_bits, cfg.fp_radius)
                if fp is None:
                    continue
            self.items.append(s)

        # Fingerprints are stored once per STRUCTURE, bit-packed. Holding one
        # float32 vector per spectrum would be ~8 KB x 2.5M = 20 GB; packed per
        # structure it is 256 bytes x 275k = 70 MB.
        self.key_index: dict[str, int] = {}
        self.fp_packed: np.ndarray | None = None
        self.item_fp: np.ndarray | None = None
        if with_targets:
            packed = []
            item_fp = np.zeros(len(self.items), dtype=np.int64)
            for i, s in enumerate(self.items):
                key = s.get("key") or s["smiles"]
                j = self.key_index.get(key)
                if j is None:
                    j = len(packed)
                    self.key_index[key] = j
                    packed.append(np.packbits(
                        morgan_bits(s["smiles"], cfg.fp_bits, cfg.fp_radius).astype(np.uint8)))
                item_fp[i] = j
            self.fp_packed = np.stack(packed) if packed else None
            self.item_fp = item_fp

        # Cleaning is the expensive part and it does not change between epochs.
        # Doing it once here turns every later epoch into pure tensor assembly.
        self.cleaned: list[tuple[np.ndarray, np.ndarray]] | None = None
        if precompute:
            self.cleaned = []
            for n, s in enumerate(self.items):
                self.cleaned.append(self._clean(s))
                if verbose and n and n % 200_000 == 0:
                    print(f"  precomputed {n}/{len(self.items)} spectra")

    def _clean(self, s: dict) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        mz, it = clean_peaks(s["mzs"], s["intensities"], s["precursor_mz"], cfg.clean)
        if mz.size > cfg.max_peaks:
            keep = np.argsort(it)[::-1][: cfg.max_peaks]
            keep.sort()
            mz, it = mz[keep], it[keep]
        return mz.astype(np.float32), it.astype(np.float32)

    def fingerprint_for(self, i: int) -> np.ndarray:
        bits = np.unpackbits(self.fp_packed[self.item_fp[i]])[: self.cfg.fp_bits]
        return bits.astype(np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        s = self.items[i]
        mz, it = self.cleaned[i] if self.cleaned is not None else self._clean(s)
        out = {
            "mzs": torch.tensor(mz, dtype=torch.float32),
            "intensities": torch.tensor(it, dtype=torch.float32),
            "precursor_mz": torch.tensor(float(s["precursor_mz"]), dtype=torch.float32),
            "adduct_id": torch.tensor(ADDUCT_TO_ID.get(s.get("adduct"), UNKNOWN_ADDUCT),
                                      dtype=torch.long),
            "mode_id": torch.tensor(0 if s.get("ionization_mode", "positive") == "positive" else 1,
                                    dtype=torch.long),
            "collision_energy": torch.tensor(float(s.get("collision_energy_ev") or 0.0),
                                             dtype=torch.float32),
            "key": s.get("key", ""),
            "molecule_id": s.get("molecule_id", ""),
        }
        if self.with_targets:
            out["fp"] = torch.from_numpy(self.fingerprint_for(i))
            out["neutral_mass"] = torch.tensor(float(s["precursor_mz"]) - 1.007276,
                                               dtype=torch.float32)
        return out


def collate(batch: list[dict]) -> dict:
    """Pad to the longest spectrum in the batch, not to max_peaks."""
    n = len(batch)
    L = max(1, max(b["mzs"].numel() for b in batch))
    mzs = torch.zeros(n, L)
    ints = torch.zeros(n, L)
    mask = torch.zeros(n, L, dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b["mzs"].numel()
        mzs[i, :k] = b["mzs"]
        ints[i, :k] = b["intensities"]
        mask[i, :k] = True
    out = {
        "mzs": mzs, "intensities": ints, "mask": mask,
        "precursor_mz": torch.stack([b["precursor_mz"] for b in batch]),
        "adduct_id": torch.stack([b["adduct_id"] for b in batch]),
        "mode_id": torch.stack([b["mode_id"] for b in batch]),
        "collision_energy": torch.stack([b["collision_energy"] for b in batch]),
        "keys": [b["key"] for b in batch],
        "molecule_ids": [b["molecule_id"] for b in batch],
    }
    if "fp" in batch[0]:
        out["fp"] = torch.stack([b["fp"] for b in batch])
        out["neutral_mass"] = torch.stack([b["neutral_mass"] for b in batch])
    return out


def pos_weight_from(dataset: SpectrumFingerprintDataset, sample: int = 20_000,
                    cap: float = 50.0) -> torch.Tensor:
    """Per-bit positive weight, so the model does not collapse to all-zeros.

    Capped, because a bit that is on in 1 spectrum out of 20,000 would otherwise
    get a weight of 20,000 and dominate the gradient with noise.
    """
    n = min(sample, len(dataset))
    idx = np.random.default_rng(0).choice(len(dataset), size=n, replace=False)
    acc = torch.zeros(dataset.cfg.fp_bits)
    for i in idx:
        acc += dataset[int(i)]["fp"]
    freq = (acc / max(n, 1)).clamp(min=1e-4)
    return ((1 - freq) / freq).clamp(max=cap)
