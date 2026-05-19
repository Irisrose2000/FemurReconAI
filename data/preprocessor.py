"""
data/preprocessor.py — CT preprocessing pipeline.

Steps
-----
1. HU windowing  → isolate bone-relevant intensities
2. Resampling    → isotropic voxels at target spacing
3. Bone mask     → threshold + morphological cleanup
4. Crop / pad    → fixed volume shape for the network
5. Normalisation → [0, 1] float32 for model input

Also contains `SyntheticFractureAugmenter` which creates artificial
fractures from intact femur masks — used to generate training pairs for
the bone-completion model without needing paired clinical data.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label as cc_label
from skimage.morphology import (
    ball, binary_closing, binary_dilation, binary_erosion
)


# ═══════════════════════════════════════════════════════════════════════════
# HU Windowing
# ═══════════════════════════════════════════════════════════════════════════

def window_hu(
    volume: np.ndarray,
    hu_min: float = 200.0,
    hu_max: float = 1800.0,
) -> np.ndarray:
    """Clip HU to bone window and normalise to [0, 1]."""
    v = np.clip(volume, hu_min, hu_max)
    v = (v - hu_min) / (hu_max - hu_min)
    return v.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Resampling
# ═══════════════════════════════════════════════════════════════════════════

def resample_volume(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.5),
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Resample *volume* so that each voxel represents *target_spacing* mm.

    Parameters
    ----------
    volume          : (D, H, W) float32 in HU
    current_spacing : (sz, sy, sx) in mm
    target_spacing  : desired (sz, sy, sx) in mm

    Returns
    -------
    resampled volume, actual_spacing (should equal target_spacing)
    """
    zoom_factors = tuple(
        cs / ts for cs, ts in zip(current_spacing, target_spacing)
    )
    resampled = ndi.zoom(volume, zoom_factors, order=1, prefilter=False)
    return resampled.astype(np.float32), target_spacing


# ═══════════════════════════════════════════════════════════════════════════
# Bone Mask Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_bone_mask(
    volume_hu: np.ndarray,
    hu_threshold: float = 300.0,
    close_radius: int = 3,
    keep_largest_n: int = 2,
) -> np.ndarray:
    """
    Return a binary mask of bone from raw HU volume.

    Strategy
    --------
    * Threshold at `hu_threshold` (cortical bone ≈ 400–1800 HU)
    * Morphological closing to fill intra-medullary canal gaps
    * Keep the N largest connected components (handles bilateral scans)

    Returns
    -------
    mask : (D, H, W) bool
    """
    raw_mask = volume_hu > hu_threshold

    # Close small holes (medullary canal, trabecular voids)
    struct = ball(close_radius)
    closed = binary_closing(raw_mask, struct)

    # Label connected components and keep the largest N
    labeled, n_comp = cc_label(closed, return_num=True)
    if n_comp == 0:
        return closed.astype(bool)

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0   # background
    top_labels = np.argsort(sizes)[::-1][:keep_largest_n]

    final_mask = np.zeros_like(closed, dtype=bool)
    for lbl in top_labels:
        final_mask |= (labeled == lbl)

    return final_mask


# ═══════════════════════════════════════════════════════════════════════════
# Crop / Pad to fixed shape
# ═══════════════════════════════════════════════════════════════════════════

def crop_or_pad(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    mask: Optional[np.ndarray] = None,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Centre-crop or zero-pad *volume* (and optional *mask*) to *target_shape*.

    Returns (cropped_volume, cropped_mask).  If mask is None → (volume, None).
    """
    result_vol = _crop_pad_array(volume, target_shape, pad_value)
    result_mask = _crop_pad_array(mask, target_shape, 0.0) if mask is not None else None
    return result_vol, result_mask


def _crop_pad_array(arr: np.ndarray, target: Tuple[int, int, int], pad_val: float):
    out = np.full(target, pad_val, dtype=arr.dtype)
    slices_src = []
    slices_dst = []
    for dim, (s, t) in enumerate(zip(arr.shape, target)):
        if s >= t:                    # crop
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:                         # pad
            start = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(start, start + s))
    out[tuple(slices_dst)] = arr[tuple(slices_src)]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Full Preprocessing Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(
    volume_hu: np.ndarray,
    spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.5),
    target_shape: Tuple[int, int, int] = (128, 128, 64),
    hu_min: float = 200.0,
    hu_max: float = 1800.0,
) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline.

    Returns a dict with keys:
        'windowed'  — normalised HU volume,   shape target_shape
        'mask'      — binary bone mask,        shape target_shape
        'spacing'   — target voxel spacing (mm)
    """
    # 1. Resample
    vol_rs, new_sp = resample_volume(volume_hu, spacing, target_spacing)

    # 2. Extract raw bone mask (before windowing, so HU values intact)
    bone_mask = extract_bone_mask(vol_rs)

    # 3. Window + normalise intensity
    vol_win = window_hu(vol_rs, hu_min, hu_max)

    # 4. Crop / pad to fixed shape
    vol_fin, mask_fin = crop_or_pad(vol_win, target_shape, mask=bone_mask.astype(np.float32))

    return {
        "windowed": vol_fin.astype(np.float32),
        "mask": (mask_fin > 0.5).astype(np.float32),
        "spacing": np.array(new_sp, dtype=np.float32),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic Fracture Augmenter
# ═══════════════════════════════════════════════════════════════════════════

class SyntheticFractureAugmenter:
    """
    Generate (fractured_mask, complete_mask) pairs from an intact femur mask.

    Strategy
    --------
    The femur shaft (diaphysis) is the most common IM-rod site.
    We simulate fractures by:

    1. Selecting a random axial plane in the diaphyseal zone (middle 60 %)
    2. Adding a random tilt (±15°) and an irregular "crack surface"
    3. Optionally removing comminuted fragments (multi-fragment fracture)
    4. Optionally introducing a gap (bone loss / comminution)

    The intact mask is the training **target**; the fractured mask is the
    model **input**.
    """

    def __init__(
        self,
        gap_range_mm: Tuple[float, float] = (2.0, 15.0),
        voxel_spacing_mm: float = 1.0,
        comminution_prob: float = 0.35,
        max_fragments: int = 3,
        seed: Optional[int] = None,
    ):
        self.gap_range = gap_range_mm
        self.vox = voxel_spacing_mm
        self.comm_prob = comminution_prob
        self.max_frags = max_fragments
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------ #

    def augment(self, complete_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Parameters
        ----------
        complete_mask : (D, H, W) binary float/bool — intact femur

        Returns
        -------
        dict with keys:
            'fractured'         — (D, H, W) float32 binary, fractured bone
            'complete'          — (D, H, W) float32 binary, original intact
            'gap_plane_z'       — axial slice index of fracture
            'gap_size_voxels'   — gap width in voxels
            'gap_size_mm'       — gap width in mm
            'fracture_type'     — 'transverse' | 'oblique' | 'comminuted'
        """
        mask = (complete_mask > 0.5).astype(bool)
        D, H, W = mask.shape

        # ── 1. Choose fracture plane in diaphysis (middle 60 %) ──────────
        z_min = int(0.20 * D)
        z_max = int(0.80 * D)
        z_plane = random.randint(z_min, z_max)

        # ── 2. Gap size ───────────────────────────────────────────────────
        gap_mm = random.uniform(*self.gap_range)
        gap_vox = max(1, int(gap_mm / self.vox))

        # ── 3. Tilt angle (oblique fracture) ─────────────────────────────
        angle_deg = random.uniform(-15, 15)
        fracture_type = "oblique" if abs(angle_deg) > 5 else "transverse"

        # ── 4. Build a tilted gap slab ────────────────────────────────────
        fractured = mask.copy()
        coords = np.indices((D, H, W))   # (3, D, H, W)
        dz = coords[0] - z_plane
        dy = coords[1] - H / 2
        dx = coords[2] - W / 2

        angle_rad = np.deg2rad(angle_deg)
        # Tilted plane: normal in the z-y plane
        plane_val = dz * np.cos(angle_rad) - dy * np.sin(angle_rad)

        # Gap slab: -half_gap <= plane_val <= +half_gap
        half = gap_vox / 2.0
        gap_mask = (plane_val >= -half) & (plane_val <= half)
        fractured[gap_mask] = False

        # ── 5. Comminution: remove extra small fragments ──────────────────
        if random.random() < self.comm_prob:
            fracture_type = "comminuted"
            n_extra = random.randint(1, self.max_frags)
            for _ in range(n_extra):
                cz = random.randint(z_min, z_max)
                cr = random.randint(3, 8)  # radius in voxels
                cy = random.randint(H // 4, 3 * H // 4)
                cx = random.randint(W // 4, 3 * W // 4)
                zz, yy, xx = np.ogrid[:D, :H, :W]
                sphere = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= cr**2
                fractured[sphere & mask] = False

        # ── 6. Add surface roughness to fracture edges ───────────────────
        noise = np.random.normal(0, 1.5, size=fractured.shape)
        smooth_noise = ndi.gaussian_filter(noise, sigma=2)
        smooth_noise = (smooth_noise - smooth_noise.min())
        smooth_noise /= smooth_noise.max() + 1e-8
        erode_mask = binary_erosion(fractured, ball(1))
        rough_edges = fractured & ~erode_mask & (smooth_noise > 0.6)
        fractured[rough_edges] = False

        return {
            "fractured": fractured.astype(np.float32),
            "complete": mask.astype(np.float32),
            "gap_plane_z": z_plane,
            "gap_size_voxels": gap_vox,
            "gap_size_mm": gap_mm,
            "fracture_type": fracture_type,
        }