"""
data/dataset.py — PyTorch Datasets for both training stages.

SegmentationDataset
    Input  : windowed CT volume   (1, D, H, W)
    Target : binary femur mask    (1, D, H, W)

BoneCompletionDataset
    Input  : fractured bone mask  (1, D, H, W)
    Target : complete bone mask   (1, D, H, W)

Both datasets support on-the-fly augmentation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from data.preprocessor import SyntheticFractureAugmenter


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(D, H, W) numpy → (1, D, H, W) float32 tensor."""
    return torch.from_numpy(arr[np.newaxis]).float()


# ── 3-D augmentation ops (all operate on (D, H, W) numpy arrays) ─────────

def random_flip(vol: np.ndarray, mask: np.ndarray):
    for ax in range(3):
        if random.random() > 0.5:
            vol  = np.flip(vol,  axis=ax).copy()
            mask = np.flip(mask, axis=ax).copy()
    return vol, mask


def random_rotate90(vol: np.ndarray, mask: np.ndarray):
    k = random.randint(0, 3)
    axes = random.choice([(0, 1), (0, 2), (1, 2)])
    vol  = np.rot90(vol,  k, axes=axes).copy()
    mask = np.rot90(mask, k, axes=axes).copy()
    return vol, mask


def random_gaussian_noise(vol: np.ndarray, sigma_range: Tuple = (0.0, 0.03)):
    sigma = random.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, vol.shape).astype(np.float32)
    return np.clip(vol + noise, 0, 1)


def random_intensity_scale(vol: np.ndarray, lo: float = 0.85, hi: float = 1.15):
    scale = random.uniform(lo, hi)
    return np.clip(vol * scale, 0, 1)


def augment_pair(vol: np.ndarray, mask: np.ndarray, augment: bool):
    if not augment:
        return vol, mask
    vol, mask = random_flip(vol, mask)
    vol, mask = random_rotate90(vol, mask)
    vol = random_gaussian_noise(vol)
    vol = random_intensity_scale(vol)
    return vol, mask


# ═══════════════════════════════════════════════════════════════════════════
# Segmentation Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SegmentationDataset(Dataset):
    """
    Expects *data_dir* to contain .npz files, each with:
        arr['windowed'] : (D, H, W) float32 — normalised CT
        arr['mask']     : (D, H, W) float32 — binary femur mask
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",        # "train" | "val" | "test"
        splits_file: Optional[str | Path] = None,
        augment: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.augment  = augment and (split == "train")

        all_files = sorted(self.data_dir.glob("*.npz"))

        # ── Load split definitions (optional) ────────────────────────────
        if splits_file and Path(splits_file).exists():
            with open(splits_file) as f:
                splits = json.load(f)
            names = set(splits.get(split, []))
            self.files = [f for f in all_files if f.stem in names]
        else:
            # Auto-split 80 / 10 / 10
            n = len(all_files)
            if split == "train":
                self.files = all_files[: int(0.8 * n)]
            elif split == "val":
                self.files = all_files[int(0.8 * n): int(0.9 * n)]
            else:
                self.files = all_files[int(0.9 * n):]

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in {data_dir} for split={split}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = np.load(self.files[idx])
        vol  = data["windowed"].astype(np.float32)
        mask = data["mask"].astype(np.float32)

        vol, mask = augment_pair(vol, mask, self.augment)

        return {
            "volume": _to_tensor(vol),   # (1, D, H, W)
            "mask":   _to_tensor(mask),  # (1, D, H, W)
            "name":   self.files[idx].stem,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Bone Completion Dataset
# ═══════════════════════════════════════════════════════════════════════════

class BoneCompletionDataset(Dataset):
    """
    Loads intact femur masks and applies SyntheticFractureAugmenter on-the-fly
    to produce (fractured, complete) training pairs.

    Each .npz must have:
        arr['mask'] : (D, H, W) float32 — binary intact femur mask
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        splits_file: Optional[str | Path] = None,
        augment: bool = True,
        fracture_augmenter: Optional[SyntheticFractureAugmenter] = None,
    ):
        self.augment = augment and (split == "train")
        self.fracture_aug = fracture_augmenter or SyntheticFractureAugmenter()

        data_dir = Path(data_dir)
        all_files = sorted(data_dir.glob("*.npz"))

        if splits_file and Path(splits_file).exists():
            with open(splits_file) as f:
                splits = json.load(f)
            names = set(splits.get(split, []))
            self.files = [f for f in all_files if f.stem in names]
        else:
            n = len(all_files)
            if split == "train":
                self.files = all_files[: int(0.8 * n)]
            elif split == "val":
                self.files = all_files[int(0.8 * n): int(0.9 * n)]
            else:
                self.files = all_files[int(0.9 * n):]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data     = np.load(self.files[idx])
        complete = data["mask"].astype(np.float32)

        # On-the-fly synthetic fracture
        aug_result = self.fracture_aug.augment(complete)
        fractured  = aug_result["fractured"]
        complete   = aug_result["complete"]

        if self.augment:
            fractured, complete = random_flip(fractured, complete)
            fractured, complete = random_rotate90(fractured, complete)

        return {
            "fractured":      _to_tensor(fractured),   # (1, D, H, W) — model INPUT
            "complete":       _to_tensor(complete),    # (1, D, H, W) — model TARGET
            "gap_mm":         torch.tensor(aug_result["gap_size_mm"], dtype=torch.float32),
            "gap_voxels":     torch.tensor(aug_result["gap_size_voxels"], dtype=torch.long),
            "fracture_type":  aug_result["fracture_type"],
            "name":           self.files[idx].stem,
        }