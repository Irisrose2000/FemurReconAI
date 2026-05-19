"""
dataset_gen/ct_simulator.py
Simulate realistic CT scan appearance from a binary femur mask.

Layers produced
---------------
  cortical bone  : 800 – 1600 HU   (dense outer shell)
  cancellous bone: 200 – 500  HU   (trabecular interior)
  medullary canal: -100 – 100 HU   (fat / marrow)
  soft tissue    : 40  – 80   HU   (muscle surround)
  background     : -1000 HU         (air)

Noise + artefacts
-----------------
  - Gaussian noise (σ ~ 20–40 HU)   simulates detector noise
  - Poisson-style intensity variation
  - Low-frequency bias field          simulates beam hardening
  - Optional ring artefact            simulates detector inhomogeneity
  - Optional motion blur (axial)      simulates patient movement

Output: (D, H, W) float32 array in Hounsfield Units,
        ready to pass into preprocessor.window_hu()
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════
# HU value ranges
# ══════════════════════════════════════════════════════════════════════

HU = dict(
    air            = -1000.0,
    fat            = -80.0,
    marrow         = 50.0,
    soft_tissue    = 60.0,
    cancellous_lo  = 200.0,
    cancellous_hi  = 500.0,
    cortical_lo    = 800.0,
    cortical_hi    = 1600.0,
)


class CTSimulator:
    """
    Convert a binary femur mask into a simulated CT HU volume.

    Parameters
    ----------
    voxel_spacing_mm  : isotropic voxel size (mm) — used for cortex erosion
    cortex_thickness_mm : approximate cortical shell thickness
    seed              : reproducibility
    """

    def __init__(
        self,
        voxel_spacing_mm:    float = 1.0,
        cortex_thickness_mm: float = 3.5,
        seed: int | None     = None,
    ):
        self.vox    = voxel_spacing_mm
        self.cortex = cortex_thickness_mm
        self.rng    = np.random.default_rng(seed)

    # ── public ──────────────────────────────────────────────────────

    def simulate(
        self,
        femur_mask:  np.ndarray,
        canal_mask:  np.ndarray | None = None,
        add_noise:          bool  = True,
        add_bias_field:     bool  = True,
        add_ring_artefact:  bool  = False,
        add_motion_blur:    bool  = False,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        femur_mask  : (D,H,W) bool — full femur voxels
        canal_mask  : (D,H,W) bool — medullary canal (optional; estimated if None)
        add_noise, add_bias_field, add_ring_artefact, add_motion_blur :
                      toggle individual artefact layers

        Returns
        -------
        ct_volume : (D,H,W) float32 in HU
        """
        D, H, W = femur_mask.shape
        rng     = self.rng

        # ── 1. Base HU volume (air everywhere) ───────────────────────
        vol = np.full((D, H, W), HU['air'], dtype=np.float32)

        # ── 2. Soft tissue surround (± 10 mm from bone) ──────────────
        bone_dilated = ndi.binary_dilation(
            femur_mask, iterations=int(10.0 / self.vox)
        )
        tissue_mask = bone_dilated & ~femur_mask
        tissue_hu   = rng.uniform(40, 80, size=(D,H,W)).astype(np.float32)
        tissue_hu  += ndi.gaussian_filter(rng.standard_normal((D,H,W)).astype(np.float32), sigma=3) * 8
        vol[tissue_mask] = tissue_hu[tissue_mask]

        # ── 3. Cancellous (trabecular) bone interior ──────────────────
        cortex_vox   = max(1, int(self.cortex / self.vox))
        cortical_shell = femur_mask & ~ndi.binary_erosion(femur_mask, iterations=cortex_vox)
        cancellous     = femur_mask & ~cortical_shell

        # Estimate canal if not provided
        if canal_mask is None:
            canal_vox  = max(1, int(cortex_vox * 2.5))
            canal_mask = ndi.binary_erosion(femur_mask, iterations=canal_vox)

        cancellous_no_canal = cancellous & ~canal_mask

        # Trabecular texture via band-pass noise
        canc_noise = self._trabecular_texture(D, H, W, rng)
        canc_hu    = (HU['cancellous_lo']
                      + canc_noise * (HU['cancellous_hi'] - HU['cancellous_lo']))
        vol[cancellous_no_canal] = canc_hu[cancellous_no_canal]

        # ── 4. Medullary canal (marrow / fat) ────────────────────────
        marrow_hu = rng.uniform(HU['fat'], HU['marrow'], size=(D,H,W)).astype(np.float32)
        marrow_hu += ndi.gaussian_filter(
            rng.standard_normal((D,H,W)).astype(np.float32), sigma=2
        ) * 15
        vol[canal_mask & femur_mask] = marrow_hu[canal_mask & femur_mask]

        # ── 5. Cortical shell ─────────────────────────────────────────
        # Dense with realistic variation: denser at mid-shaft, less at ends
        cortex_base = rng.uniform(HU['cortical_lo'], HU['cortical_hi'],
                                  size=(D,H,W)).astype(np.float32)
        # Low-frequency depth variation
        depth_map = ndi.gaussian_filter(
            rng.standard_normal((D,H,W)).astype(np.float32), sigma=6
        ) * 80
        cortex_hu = cortex_base + depth_map
        cortex_hu = np.clip(cortex_hu, HU['cortical_lo'] - 100, HU['cortical_hi'] + 100)
        vol[cortical_shell] = cortex_hu[cortical_shell]

        # ── 6. Gaussian noise ─────────────────────────────────────────
        if add_noise:
            sigma_noise = rng.uniform(18, 40)
            noise       = rng.standard_normal((D,H,W)).astype(np.float32) * sigma_noise
            # More noise in soft tissue / air, less in dense bone
            noise_scale       = np.ones((D,H,W), dtype=np.float32)
            noise_scale[femur_mask] *= 0.4   # bone is denser → less noise
            vol += noise * noise_scale

        # ── 7. Bias field (beam hardening) ────────────────────────────
        if add_bias_field:
            vol = self._apply_bias_field(vol, rng)

        # ── 8. Ring artefact ──────────────────────────────────────────
        if add_ring_artefact:
            vol = self._apply_ring_artefact(vol, rng)

        # ── 9. Motion blur (axial) ────────────────────────────────────
        if add_motion_blur:
            blur_sigma = rng.uniform(0.3, 1.2)
            vol        = ndi.gaussian_filter(vol, sigma=[blur_sigma, 0, 0])

        return vol.astype(np.float32)

    # ── private helpers ──────────────────────────────────────────────

    def _trabecular_texture(self, D, H, W, rng) -> np.ndarray:
        """
        Multi-scale noise simulating trabecular bone texture.
        Combines coarse (bone islands) + fine (trabeculae) scales.
        """
        coarse = ndi.gaussian_filter(
            rng.standard_normal((D,H,W)).astype(np.float32), sigma=4
        )
        fine   = ndi.gaussian_filter(
            rng.standard_normal((D,H,W)).astype(np.float32), sigma=1.2
        )
        texture = 0.7 * coarse + 0.3 * fine
        # Normalise to [0, 1]
        lo, hi = texture.min(), texture.max()
        return (texture - lo) / (hi - lo + 1e-8)

    def _apply_bias_field(self, vol: np.ndarray, rng) -> np.ndarray:
        """
        Simulate beam-hardening bias field: smooth low-frequency intensity
        gradient across the volume (up to ±80 HU variation).
        """
        D, H, W = vol.shape
        # Small random grid, upsampled to full size
        grid_size = 4
        small = rng.standard_normal((grid_size, grid_size, grid_size)).astype(np.float32)
        bias  = ndi.zoom(small, [D/grid_size, H/grid_size, W/grid_size], order=1)
        # Trim/pad to exact shape
        bias  = bias[:D, :H, :W]
        scale = rng.uniform(30, 80)
        return vol + bias * scale

    def _apply_ring_artefact(self, vol: np.ndarray, rng) -> np.ndarray:
        """
        Add concentric ring artefacts (simulate detector inhomogeneity).
        Affects axial (H-W) slices only.
        """
        D, H, W = vol.shape
        cy, cx  = H / 2.0, W / 2.0
        yy, xx  = np.mgrid[0:H, 0:W].astype(np.float32)
        radius  = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        n_rings = rng.integers(2, 5)
        for _ in range(n_rings):
            r0        = rng.uniform(10, min(H, W) * 0.45)
            width     = rng.uniform(0.5, 2.0)
            amplitude = rng.uniform(10, 40) * rng.choice([-1, 1])
            ring_mask = (radius >= r0) & (radius <= r0 + width)
            vol[:, ring_mask] += amplitude
        return vol