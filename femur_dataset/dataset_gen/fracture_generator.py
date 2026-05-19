"""
dataset_gen/fracture_generator.py
Applies realistic fractures to an intact femur voxel mask.

Fracture types implemented (AO/OTA inspired)
--------------------------------------------
  A1  — Simple transverse       (flat plane, <30° tilt)
  A2  — Simple oblique          (tilted plane, 30–60°)
  A3  — Simple spiral           (helical surface)
  B1  — Wedge                   (2 cuts → butterfly/wedge fragment)
  B2  — Comminuted wedge        (wedge + extra fragments)
  C1  — Segmental               (2 separate fracture levels)
  C2  — Highly comminuted       (3+ irregular cuts)

Each fracture type returns:
  fractured_mask  — what remains (voxels of bone present)
  complete_mask   — original intact mask (training target)
  gap_mask        — missing bone (complete XOR fractured)
  metadata        — fracture type, gap size, zone, fragment count, etc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import ball, binary_erosion


# ══════════════════════════════════════════════════════════════════════
# Result container
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FractureResult:
    fractured_mask:  np.ndarray          # (D,H,W) bool — what remains
    complete_mask:   np.ndarray          # (D,H,W) bool — intact (target)
    gap_mask:        np.ndarray          # (D,H,W) bool — removed voxels
    ao_code:         str                 # A1 / A2 / A3 / B1 / B2 / C1 / C2
    fracture_type:   str                 # transverse / oblique / spiral / etc.
    gap_size_mm:     float               # primary gap width in mm
    fracture_z:      int                 # primary fracture axial slice
    n_fragments:     int
    fragment_zones:  List[str] = field(default_factory=list)
    tilt_angle_deg:  float = 0.0


# ══════════════════════════════════════════════════════════════════════
# Plane / surface helpers
# ══════════════════════════════════════════════════════════════════════

def _tilted_plane_slab(
    shape: Tuple[int,int,int],
    z_centre: float,
    half_gap: float,
    tilt_xz_deg: float = 0.0,
    tilt_yz_deg: float = 0.0,
) -> np.ndarray:
    """
    Return a boolean slab mask representing the fracture gap.
    The slab is centred at z_centre with half-width half_gap,
    optionally tilted in the XZ and YZ planes.
    """
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    dz = zz - z_centre
    dx = xx - W / 2.0
    dy = yy - H / 2.0

    tx = np.deg2rad(tilt_xz_deg)
    ty = np.deg2rad(tilt_yz_deg)

    plane_val = (dz * np.cos(tx) * np.cos(ty)
                 - dx * np.sin(tx)
                 - dy * np.sin(ty))

    return (plane_val >= -half_gap) & (plane_val <= half_gap)


def _spiral_surface(
    shape: Tuple[int,int,int],
    z_centre: float,
    half_gap: float,
    pitch: float = 0.15,
    amplitude: float = 8.0,
) -> np.ndarray:
    """
    Helical (spiral) fracture gap.
    The fracture plane rotates around the Z-axis as it travels along Z.
    """
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)
    dz = zz - z_centre
    dx = xx - W / 2.0
    dy = yy - H / 2.0

    # Rotation angle proportional to z position
    theta = pitch * dz
    rotated_x = dx * np.cos(theta) - dy * np.sin(theta)

    plane_val = dz * np.cos(np.deg2rad(20)) - rotated_x * np.sin(np.deg2rad(20)) * amplitude / 10.0
    return (plane_val >= -half_gap) & (plane_val <= half_gap)


def _add_surface_roughness(gap_mask: np.ndarray, sigma: float = 1.5, threshold: float = 0.4) -> np.ndarray:
    """Add irregular jagged edges to a fracture gap to mimic real bone breaks."""
    noise = np.random.randn(*gap_mask.shape).astype(np.float32)
    smooth = ndi.gaussian_filter(noise, sigma=sigma)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
    # Dilate the gap slightly where noise is high
    extra = ndi.binary_dilation(gap_mask, iterations=1) & (smooth > (1 - threshold))
    return gap_mask | extra


def _sphere_fragment(
    shape: Tuple[int,int,int],
    cz: float, cy: float, cx: float,
    radius: float,
) -> np.ndarray:
    """Spherical fragment removal region."""
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    return ((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2) <= radius**2


# ══════════════════════════════════════════════════════════════════════
# Main fracture generator
# ══════════════════════════════════════════════════════════════════════

class FractureGenerator:
    """
    Apply synthetic fractures to an intact femur mask.

    Parameters
    ----------
    voxel_spacing_mm  : physical size of each voxel (mm) — used to
                        convert gap sizes from mm to voxels
    seed              : random seed
    """

    # Fracture zone: restrict to diaphysis (middle portion of shaft)
    DIAPHYSIS_ZONE = (0.25, 0.78)

    def __init__(
        self,
        voxel_spacing_mm: float = 1.0,
        seed: int | None = None,
    ):
        self.vox = voxel_spacing_mm
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ── public ──────────────────────────────────────────────────────

    def apply(
        self,
        intact_mask: np.ndarray,
        fracture_type: str | None = None,
        diaphysis_range: Tuple[int, int] | None = None,
    ) -> FractureResult:
        """
        Parameters
        ----------
        intact_mask      : (D,H,W) bool — intact femur
        fracture_type    : 'A1'|'A2'|'A3'|'B1'|'B2'|'C1'|'C2'|None
                           None → random selection weighted toward common types
        diaphysis_range  : (z_top, z_bot) voxel indices of the shaft.
                           If None, inferred from DIAPHYSIS_ZONE fraction.

        Returns
        -------
        FractureResult
        """
        D, H, W = intact_mask.shape
        mask    = intact_mask.astype(bool)

        # Infer diaphysis range
        if diaphysis_range is None:
            z0 = int(self.DIAPHYSIS_ZONE[0] * D)
            z1 = int(self.DIAPHYSIS_ZONE[1] * D)
        else:
            z0, z1 = diaphysis_range

        # Choose fracture type (weighted — transverse/oblique most common)
        if fracture_type is None:
            fracture_type = random.choices(
                ['A1', 'A2', 'A3', 'B1', 'B2', 'C1', 'C2'],
                weights=[0.25, 0.20, 0.10, 0.15, 0.12, 0.10, 0.08],
            )[0]

        dispatch = {
            'A1': self._transverse,
            'A2': self._oblique,
            'A3': self._spiral,
            'B1': self._wedge,
            'B2': self._comminuted_wedge,
            'C1': self._segmental,
            'C2': self._highly_comminuted,
        }
        fn = dispatch.get(fracture_type, self._transverse)
        result = fn(mask, z0, z1, D, H, W)
        return result

    # ── fracture implementations ─────────────────────────────────────

    def _transverse(self, mask, z0, z1, D, H, W) -> FractureResult:
        z_c    = random.randint(z0, z1)
        gap_mm = random.uniform(2.0, 10.0)
        half   = (gap_mm / self.vox) / 2.0
        tilt   = random.uniform(-12, 12)

        gap = _tilted_plane_slab((D,H,W), z_c, half, tilt_xz_deg=tilt)
        gap = _add_surface_roughness(gap, sigma=1.2, threshold=0.35)
        gap = gap & mask

        return FractureResult(
            fractured_mask = mask & ~gap,
            complete_mask  = mask,
            gap_mask       = gap,
            ao_code        = 'A1',
            fracture_type  = 'transverse',
            gap_size_mm    = gap_mm,
            fracture_z     = z_c,
            n_fragments    = 2,
            tilt_angle_deg = abs(tilt),
        )

    def _oblique(self, mask, z0, z1, D, H, W) -> FractureResult:
        z_c    = random.randint(z0, z1)
        gap_mm = random.uniform(3.0, 14.0)
        half   = (gap_mm / self.vox) / 2.0
        tilt_xz = random.choice([-1, 1]) * random.uniform(30, 60)
        tilt_yz = random.uniform(-15, 15)

        gap = _tilted_plane_slab((D,H,W), z_c, half, tilt_xz, tilt_yz)
        gap = _add_surface_roughness(gap, sigma=1.5, threshold=0.3)
        gap = gap & mask

        return FractureResult(
            fractured_mask = mask & ~gap,
            complete_mask  = mask,
            gap_mask       = gap,
            ao_code        = 'A2',
            fracture_type  = 'oblique',
            gap_size_mm    = gap_mm,
            fracture_z     = z_c,
            n_fragments    = 2,
            tilt_angle_deg = abs(tilt_xz),
        )

    def _spiral(self, mask, z0, z1, D, H, W) -> FractureResult:
        z_c    = random.randint(z0, z1)
        gap_mm = random.uniform(3.0, 12.0)
        half   = (gap_mm / self.vox) / 2.0
        pitch  = random.uniform(0.10, 0.22)

        gap = _spiral_surface((D,H,W), z_c, half, pitch=pitch)
        gap = _add_surface_roughness(gap, sigma=2.0, threshold=0.30)
        gap = gap & mask

        return FractureResult(
            fractured_mask = mask & ~gap,
            complete_mask  = mask,
            gap_mask       = gap,
            ao_code        = 'A3',
            fracture_type  = 'spiral',
            gap_size_mm    = gap_mm,
            fracture_z     = z_c,
            n_fragments    = 2,
            tilt_angle_deg = 45.0,
        )

    def _wedge(self, mask, z0, z1, D, H, W) -> FractureResult:
        z_c    = random.randint(z0 + 5, z1 - 5)
        gap_mm = random.uniform(4.0, 16.0)
        half   = (gap_mm / self.vox) / 2.0
        spread = random.uniform(1.5, 3.0)   # wedge width multiplier

        # Two angled cuts forming a wedge
        gap1 = _tilted_plane_slab((D,H,W), z_c - half*0.6, half*0.5, tilt_xz_deg=random.uniform(15,35))
        gap2 = _tilted_plane_slab((D,H,W), z_c + half*0.6, half*0.5, tilt_xz_deg=random.uniform(-35,-15))
        gap  = (gap1 | gap2)
        gap  = _add_surface_roughness(gap, sigma=1.5, threshold=0.32)
        gap  = gap & mask

        return FractureResult(
            fractured_mask = mask & ~gap,
            complete_mask  = mask,
            gap_mask       = gap,
            ao_code        = 'B1',
            fracture_type  = 'wedge',
            gap_size_mm    = gap_mm,
            fracture_z     = z_c,
            n_fragments    = 3,
            tilt_angle_deg = 25.0,
        )

    def _comminuted_wedge(self, mask, z0, z1, D, H, W) -> FractureResult:
        # Start with a wedge
        base   = self._wedge(mask, z0, z1, D, H, W)
        gap    = base.gap_mask.copy()

        # Add 1-2 small spherical fragment removals
        n_extra = random.randint(1, 2)
        cy, cx  = H / 2.0, W / 2.0
        for _ in range(n_extra):
            fz = base.fracture_z + random.randint(-8, 8)
            fr = random.uniform(3.0, 7.0)
            fy = cy + random.uniform(-0.12, 0.12) * H
            fx = cx + random.uniform(-0.12, 0.12) * W
            gap |= _sphere_fragment((D,H,W), fz, fy, fx, fr) & mask

        frac = mask & ~gap
        return FractureResult(
            fractured_mask = frac,
            complete_mask  = mask,
            gap_mask       = gap,
            ao_code        = 'B2',
            fracture_type  = 'comminuted_wedge',
            gap_size_mm    = base.gap_size_mm,
            fracture_z     = base.fracture_z,
            n_fragments    = 3 + n_extra,
            tilt_angle_deg = 25.0,
        )

    def _segmental(self, mask, z0, z1, D, H, W) -> FractureResult:
        """Two separate fracture levels — creates a floating middle segment."""
        span     = z1 - z0
        z_upper  = z0 + int(span * random.uniform(0.20, 0.38))
        z_lower  = z0 + int(span * random.uniform(0.62, 0.80))

        gap_mm1  = random.uniform(2.0, 8.0)
        gap_mm2  = random.uniform(2.0, 8.0)
        half1    = (gap_mm1 / self.vox) / 2.0
        half2    = (gap_mm2 / self.vox) / 2.0

        gap1 = _tilted_plane_slab((D,H,W), z_upper, half1, tilt_xz_deg=random.uniform(-20,20))
        gap2 = _tilted_plane_slab((D,H,W), z_lower, half2, tilt_xz_deg=random.uniform(-20,20))
        gap  = (gap1 | gap2)
        gap  = _add_surface_roughness(gap, sigma=1.3, threshold=0.30)
        gap  = gap & mask

        return FractureResult(
            fractured_mask = mask & ~gap,
            complete_mask  = mask,
            gap_mask       = gap,
            ao_code        = 'C1',
            fracture_type  = 'segmental',
            gap_size_mm    = (gap_mm1 + gap_mm2) / 2.0,
            fracture_z     = (z_upper + z_lower) // 2,
            n_fragments    = 3,
            tilt_angle_deg = 15.0,
        )

    def _highly_comminuted(self, mask, z0, z1, D, H, W) -> FractureResult:
        """Multiple fracture planes + spherical fragment removals."""
        z_c   = random.randint(z0 + 10, z1 - 10)
        gap_mm = random.uniform(15.0, 30.0)
        half   = (gap_mm / self.vox) / 2.0

        # Main wide gap
        gap = _tilted_plane_slab((D,H,W), z_c, half, tilt_xz_deg=random.uniform(-25,25))

        # 3–5 extra irregular spherical removals within the gap zone
        n_extra = random.randint(3, 5)
        cy, cx  = H / 2.0, W / 2.0
        for _ in range(n_extra):
            fz = z_c + random.uniform(-half * 0.7, half * 0.7)
            fr = random.uniform(2.5, 6.0)
            fy = cy + random.uniform(-0.15, 0.15) * H
            fx = cx + random.uniform(-0.15, 0.15) * W
            gap |= _sphere_fragment((D,H,W), fz, fy, fx, fr)

        gap = _add_surface_roughness(gap, sigma=2.0, threshold=0.28)
        gap = gap & mask

        return FractureResult(
            fractured_mask = mask & ~gap,
            complete_mask  = mask,
            gap_mask       = gap,
            ao_code        = 'C2',
            fracture_type  = 'highly_comminuted',
            gap_size_mm    = gap_mm,
            fracture_z     = z_c,
            n_fragments    = 2 + n_extra,
            tilt_angle_deg = 20.0,
        )