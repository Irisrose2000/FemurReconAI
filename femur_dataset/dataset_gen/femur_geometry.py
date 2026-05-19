"""
dataset_gen/femur_geometry.py
Procedural 3-D femur bone generator.

Produces a binary voxel mask that approximates the real femur anatomy:
  - Femoral head (sphere)
  - Femoral neck (tapered cylinder, angled)
  - Greater & lesser trochanters (ellipsoid bumps)
  - Diaphysis / shaft (hollow cylinder with medullary canal)
  - Distal metaphysis & condyles (widened bicondylar end)

All shapes are parameterised with randomised variation so every
generated sample looks slightly different — simulating patient-to-patient
anatomical variability.

Output: (D, H, W) binary numpy array  (dtype=bool)
        voxel spacing assumed 1 mm isotropic
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════

def _ellipsoid(grid, cx, cy, cz, rx, ry, rz, angle_xz=0.0):
    """Return True for voxels inside a (possibly rotated) ellipsoid."""
    zz, yy, xx = grid
    dz = zz - cz
    dy = yy - cy
    dx = xx - cx
    # rotate in x-z plane
    if angle_xz != 0.0:
        cos_a, sin_a = np.cos(angle_xz), np.sin(angle_xz)
        dx2 =  cos_a * dx + sin_a * dz
        dz2 = -sin_a * dx + cos_a * dz
        dx, dz = dx2, dz2
    return (dx/rx)**2 + (dy/ry)**2 + (dz/rz)**2 <= 1.0


def _cylinder(grid, ax_z0, ax_z1, cx, cy, rx, ry, taper=0.0):
    """
    Axis-aligned (Z-axis) tapered cylinder between z0 and z1.
    taper > 0 → top radius larger; taper < 0 → top radius smaller.
    """
    zz, yy, xx = grid
    mask_z = (zz >= ax_z0) & (zz <= ax_z1)
    t  = np.clip((zz - ax_z0) / max(ax_z1 - ax_z0, 1), 0, 1)
    rx_t = rx * (1 + taper * t)
    ry_t = ry * (1 + taper * t)
    inside = ((xx - cx)**2 / rx_t**2 + (yy - cy)**2 / ry_t**2) <= 1.0
    return mask_z & inside


# ══════════════════════════════════════════════════════════════════════
# main generator
# ══════════════════════════════════════════════════════════════════════

class FemurGeometryGenerator:
    """
    Parameters
    ----------
    volume_shape : (D, H, W) voxel grid size
    seed         : random seed for reproducibility
    left_side    : True → left femur (mirror image)
    """

    def __init__(
        self,
        volume_shape: Tuple[int,int,int] = (200, 80, 80),
        seed: int | None = None,
        left_side: bool = False,
    ):
        self.shape    = volume_shape
        self.rng      = np.random.default_rng(seed)
        self.left     = left_side

    # ── public ──────────────────────────────────────────────────────

    def generate(self) -> dict:
        """
        Returns
        -------
        dict with keys:
          'mask'        : (D,H,W) bool  — binary femur voxels
          'canal_mask'  : (D,H,W) bool  — medullary canal only
          'params'      : dict of anatomical measurements (mm)
        """
        D, H, W = self.shape
        rng = self.rng

        # ── anatomical parameters (with realistic variation) ──────────
        shaft_len   = rng.uniform(0.55, 0.65) * D      # diaphysis length
        head_r      = rng.uniform(0.11, 0.14) * H      # femoral head radius
        neck_angle  = rng.uniform(0.18, 0.28)           # neck-shaft angle (rad)
        neck_len    = rng.uniform(0.14, 0.20) * D
        shaft_r     = rng.uniform(0.09, 0.13) * H      # shaft radius
        canal_r     = rng.uniform(0.045, 0.07) * H     # medullary canal radius
        distal_r    = rng.uniform(0.14, 0.18) * H      # distal condyle radius
        cortex_th   = rng.uniform(0.025, 0.040) * H    # cortex thickness

        # centre of volume
        cx = W / 2.0 + rng.uniform(-2, 2)
        cy = H / 2.0 + rng.uniform(-2, 2)

        # z-positions (0=top=proximal, D=distal)
        z_head      = rng.uniform(0.07, 0.12) * D
        z_neck_end  = z_head  + neck_len
        z_shaft_top = z_neck_end
        z_shaft_bot = z_shaft_top + shaft_len
        z_distal    = D - rng.uniform(0.05, 0.10) * D

        # ── build grid ────────────────────────────────────────────────
        grid = np.mgrid[0:D, 0:H, 0:W].astype(np.float32)

        # ── femoral head ─────────────────────────────────────────────
        # offset laterally from shaft centre
        head_offset_x = (1 if self.left else -1) * rng.uniform(0.12, 0.18) * W
        head_cx = cx + head_offset_x
        head_cy = cy + rng.uniform(-0.04, 0.04) * H

        head = _ellipsoid(
            grid, head_cx, head_cy, z_head,
            head_r, head_r * rng.uniform(0.9, 1.1), head_r * 0.9,
        )

        # ── femoral neck ──────────────────────────────────────────────
        neck_r   = head_r * rng.uniform(0.55, 0.70)
        neck_taper = rng.uniform(-0.25, -0.10)   # narrows toward shaft
        neck = _cylinder(
            grid, z_head, z_neck_end,
            head_cx, head_cy, neck_r, neck_r, taper=neck_taper,
        )

        # ── greater trochanter ────────────────────────────────────────
        gt_offset_x = (1 if not self.left else -1) * rng.uniform(0.12, 0.18) * W
        gt_r = rng.uniform(0.08, 0.11) * H
        gt = _ellipsoid(
            grid,
            cx + gt_offset_x,
            cy,
            z_neck_end + rng.uniform(-0.01, 0.02) * D,
            gt_r * 0.7, gt_r * 0.9, gt_r,
        )

        # ── lesser trochanter ─────────────────────────────────────────
        lt_r = rng.uniform(0.04, 0.07) * H
        lt_z = z_neck_end + rng.uniform(0.02, 0.05) * D
        lt_y = cy + rng.uniform(0.06, 0.10) * H
        lt = _ellipsoid(grid, cx, lt_y, lt_z, lt_r * 0.6, lt_r, lt_r * 0.7)

        # ── shaft (hollow) ────────────────────────────────────────────
        shaft_taper = rng.uniform(0.05, 0.15)     # slightly wider distally
        shaft_outer = _cylinder(
            grid, z_shaft_top, z_shaft_bot, cx, cy, shaft_r, shaft_r, taper=shaft_taper,
        )
        # medullary canal
        canal_outer = _cylinder(
            grid, z_shaft_top, z_shaft_bot, cx, cy, canal_r, canal_r,
            taper=shaft_taper * 0.5,
        )
        shaft = shaft_outer & ~canal_outer

        # ── distal metaphysis & condyles ──────────────────────────────
        distal_taper = rng.uniform(0.30, 0.50)
        distal = _cylinder(
            grid, z_shaft_bot, z_distal + distal_r, cx, cy,
            shaft_r * (1 + distal_taper), shaft_r * (1 + distal_taper * 0.8),
            taper=0,
        )

        # medial condyle
        mc_r = distal_r * rng.uniform(0.85, 1.0)
        mc_x = cx + (1 if self.left else -1) * rng.uniform(0.05, 0.10) * W
        mc = _ellipsoid(grid, mc_x, cy - rng.uniform(0.04,0.08)*H, z_distal,
                        mc_r*0.9, mc_r*0.85, mc_r)

        # lateral condyle
        lc_r = distal_r * rng.uniform(0.85, 1.0)
        lc_x = cx - (1 if self.left else -1) * rng.uniform(0.05, 0.10) * W
        lc = _ellipsoid(grid, lc_x, cy + rng.uniform(0.04,0.08)*H, z_distal,
                        lc_r*0.9, lc_r*0.85, lc_r)

        # ── combine ───────────────────────────────────────────────────
        femur = head | neck | gt | lt | shaft | distal | mc | lc

        # ── smooth borders (remove voxellation artefacts) ─────────────
        femur_f = gaussian_filter(femur.astype(np.float32), sigma=1.0)
        femur   = femur_f > 0.35

        # canal mask (for measurement)
        canal_f = gaussian_filter(canal_outer.astype(np.float32), sigma=0.8)
        canal   = canal_f > 0.3

        params = {
            "head_radius_mm":    round(float(head_r), 2),
            "neck_length_mm":    round(float(neck_len), 2),
            "shaft_radius_mm":   round(float(shaft_r), 2),
            "canal_radius_mm":   round(float(canal_r), 2),
            "cortex_thickness_mm": round(float(cortex_th), 2),
            "distal_radius_mm":  round(float(distal_r), 2),
            "z_shaft_top":       round(float(z_shaft_top), 1),
            "z_shaft_bot":       round(float(z_shaft_bot), 1),
            "z_distal":          round(float(z_distal), 1),
            "femur_length_mm":   round(float(z_distal - z_head), 1),
            "canal_diameter_mm": round(float(canal_r * 2), 2),
            "isthmus_diameter_mm": round(float(canal_r * 2 * 0.92), 2),
        }

        return {
            "mask":       femur,
            "canal_mask": canal,
            "params":     params,
            "z_diaphysis_range": (int(z_shaft_top), int(z_shaft_bot)),
        }