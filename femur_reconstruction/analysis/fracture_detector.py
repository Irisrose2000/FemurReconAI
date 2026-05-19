"""
analysis/fracture_detector.py — Fracture classification & IM rod sizing.

Takes the fractured bone mask and (optionally) the completed mask to:

1.  Count and label bone fragments
2.  Classify the fracture pattern (AO/OTA classification)
3.  Measure the medullary canal for IM rod sizing
4.  Suggest IM rod diameter and length
5.  Detect the fracture plane location and orientation

AO/OTA Fracture Classification (simplified)
-------------------------------------------
  A1 — Simple transverse          (single fracture line, <30° from perpendicular)
  A2 — Simple oblique             (single fracture line, ≥30° from perpendicular)
  A3 — Simple spiral              (helical fracture plane)
  B1 — Wedge (intact wedge)       (main fragments + wedge piece)
  B2 — Wedge (comminuted wedge)
  B3 — Complex (multi-fragmentary)
  C1 — Segmental                  (two distinct fracture levels)
  C2 — Segmental + comminuted
  C3 — Irregular (highly comminuted, >4 fragments)

IM Rod Sizing Reference (standard clinical ranges)
---------------------------------------------------
  Diameter : 9, 10, 11, 12, 13 mm  (based on canal isthmus diameter)
  Length   : 340–480 mm in 20 mm steps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.measure import label as cc_label, regionprops


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BoneFragment:
    id:            int
    volume_mm3:    float
    centroid_mm:   Tuple[float, float, float]
    bbox_voxels:   Tuple
    is_main:       bool     # True for the two largest (proximal/distal)
    label:         str      # "proximal" | "distal" | "fragment_N"


@dataclass
class FractureDetectionResult:
    # ── Fragment info ──────────────────────────────────────────────────────
    n_fragments:        int
    fragments:          List[BoneFragment] = field(default_factory=list)

    # ── AO classification ─────────────────────────────────────────────────
    ao_code:            str = "unknown"
    ao_description:     str = ""
    fracture_pattern:   str = ""    # transverse / oblique / spiral / comminuted / segmental

    # ── Fracture plane ────────────────────────────────────────────────────
    fracture_z_voxel:   Optional[int]   = None    # primary axial slice
    fracture_z_mm:      Optional[float] = None
    fracture_angle_deg: float = 0.0               # tilt from axial plane

    # ── Medullary canal measurements ──────────────────────────────────────
    canal_diameter_mm:  float = 0.0
    canal_measurements: Dict[str, float] = field(default_factory=dict)

    # ── IM Rod recommendation ─────────────────────────────────────────────
    recommended_rod_diameter_mm: Optional[float] = None
    recommended_rod_length_mm:   Optional[float] = None
    rod_sizing_confidence:        str = "low"     # low / medium / high


# ═══════════════════════════════════════════════════════════════════════════
# IM Rod Size Tables
# ═══════════════════════════════════════════════════════════════════════════

ROD_DIAMETERS_MM = [9.0, 10.0, 11.0, 12.0, 13.0]
ROD_LENGTHS_MM   = list(range(340, 500, 20))   # 340, 360, … 480

def select_rod_diameter(canal_mm: float) -> Tuple[float, str]:
    """
    Standard rule: rod diameter = canal isthmus diameter - 1 mm
    (allows 0.5 mm cortex clearance on each side).
    """
    target = canal_mm - 1.0
    if target < ROD_DIAMETERS_MM[0]:
        return ROD_DIAMETERS_MM[0], "low"
    if target > ROD_DIAMETERS_MM[-1]:
        return ROD_DIAMETERS_MM[-1], "medium"
    # Round to nearest available diameter
    closest = min(ROD_DIAMETERS_MM, key=lambda d: abs(d - target))
    conf    = "high" if abs(closest - target) <= 0.5 else "medium"
    return closest, conf


def select_rod_length(femur_length_mm: float) -> float:
    """Rod length ≈ femur length − 20 mm (for distal lock clearance)."""
    target = femur_length_mm - 20.0
    return min(ROD_LENGTHS_MM, key=lambda l: abs(l - target))


# ═══════════════════════════════════════════════════════════════════════════
# Canal Measurement
# ═══════════════════════════════════════════════════════════════════════════

def measure_medullary_canal(
    bone_mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    isthmus_zone: Tuple[float, float] = (0.35, 0.55),
) -> Dict[str, float]:
    """
    Estimate the medullary canal diameter from the bone mask.

    Method
    ------
    1. In the isthmus zone (middle ~35–55% of the shaft), erode the outer
       cortex by 2 mm to expose the canal boundary.
    2. For each axial slice, compute the equivalent circle diameter of the
       bone cross-section minus the estimated cortical thickness (≈3 mm).
    3. Return the minimum diameter (narrowest point = rod constraint).

    Returns dict with:
        isthmus_diameter_mm   — narrowest canal diameter
        mean_canal_mm         — mean canal diameter in isthmus zone
        femur_length_mm       — estimated total femur length
    """
    D, H, W  = bone_mask.shape
    sp_z, sp_y, sp_x = spacing_mm

    z0 = int(isthmus_zone[0] * D)
    z1 = int(isthmus_zone[1] * D)

    canal_diameters = []
    for z in range(z0, z1):
        sl = bone_mask[z]
        if sl.sum() == 0:
            continue
        area_mm2 = float(sl.sum()) * sp_y * sp_x
        # Equivalent circle diameter from cross-section area
        import math
        outer_d = 2.0 * math.sqrt(area_mm2 / math.pi)
        cortex  = 4.0          # assume 2 mm cortex each side
        canal_d = max(outer_d - cortex, 3.0)  # minimum 3 mm canal
        canal_diameters.append(canal_d)

    if not canal_diameters:
        return {"isthmus_diameter_mm": 10.0, "mean_canal_mm": 10.0, "femur_length_mm": 400.0}

    # Femur length = axial extent of the mask
    z_indices = np.where(bone_mask.any(axis=(1, 2)))[0]
    femur_len = float(len(z_indices)) * sp_z

    return {
        "isthmus_diameter_mm": round(float(np.min(canal_diameters)), 2),
        "mean_canal_mm":       round(float(np.mean(canal_diameters)), 2),
        "femur_length_mm":     round(femur_len, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# AO Classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_ao(
    n_fragments: int,
    fracture_angle: float,
    n_fracture_levels: int,
) -> Tuple[str, str, str]:
    """Return (ao_code, ao_description, pattern)."""

    if n_fragments <= 2:
        # Simple fracture (A-type)
        if abs(fracture_angle) < 30:
            return "A1", "Simple transverse fracture", "transverse"
        elif abs(fracture_angle) < 60:
            return "A2", "Simple oblique fracture", "oblique"
        else:
            return "A3", "Simple spiral fracture", "spiral"

    elif n_fragments == 3:
        if abs(fracture_angle) < 45:
            return "B1", "Wedge fracture with intact wedge", "wedge"
        else:
            return "B2", "Wedge fracture with comminuted wedge", "comminuted"

    elif n_fracture_levels >= 2:
        if n_fragments <= 4:
            return "C1", "Segmental fracture", "segmental"
        else:
            return "C2", "Segmental fracture with comminution", "comminuted"

    else:
        if n_fragments <= 5:
            return "B3", "Complex multifragmentary fracture", "comminuted"
        else:
            return "C3", "Highly comminuted irregular fracture", "comminuted"


# ═══════════════════════════════════════════════════════════════════════════
# Main Detector
# ═══════════════════════════════════════════════════════════════════════════

class FractureDetector:
    """
    Parameters
    ----------
    spacing_mm             : (sz, sy, sx) voxel spacing in mm
    min_fragment_vol_mm3   : ignore components smaller than this
    """

    def __init__(
        self,
        spacing_mm:           Tuple[float, float, float] = (1.0, 1.0, 1.0),
        min_fragment_vol_mm3: float = 500.0,
    ):
        self.spacing      = spacing_mm
        self.voxel_vol    = float(np.prod(spacing_mm))
        self.min_frag_vol = min_fragment_vol_mm3

    def detect(
        self,
        fractured_mask: np.ndarray,
        completed_mask: Optional[np.ndarray] = None,
    ) -> FractureDetectionResult:
        """
        Parameters
        ----------
        fractured_mask : (D, H, W) — physically present bone
        completed_mask : (D, H, W) — reconstructed intact bone (used for length/canal)
        """
        frac = (fractured_mask > 0.5).astype(bool)
        ref  = (completed_mask  > 0.5).astype(bool) if completed_mask is not None else frac

        # ── 1. Fragment labeling ──────────────────────────────────────────
        labeled, n_comp = cc_label(frac, return_num=True)
        props = regionprops(labeled)

        fragments = []
        for prop in props:
            vol = prop.area * self.voxel_vol
            if vol < self.min_frag_vol:
                continue
            cz, cy, cx = prop.centroid
            fragments.append(BoneFragment(
                id          = prop.label,
                volume_mm3  = vol,
                centroid_mm = (
                    cz * self.spacing[0],
                    cy * self.spacing[1],
                    cx * self.spacing[2],
                ),
                bbox_voxels = prop.bbox,
                is_main     = False,
                label       = "fragment",
            ))

        fragments.sort(key=lambda f: f.volume_mm3, reverse=True)
        for i, f in enumerate(fragments):
            if i == 0:
                f.label  = "proximal"
                f.is_main = True
            elif i == 1:
                f.label  = "distal"
                f.is_main = True
            else:
                f.label  = f"fragment_{i - 1}"
            f.id = i + 1

        # ── 2. Fracture plane detection ───────────────────────────────────
        frac_z, frac_z_mm, frac_angle = self._detect_fracture_plane(frac, fragments)

        # ── 3. Canal measurement on the COMPLETED bone (more reliable) ────
        canal_meas = measure_medullary_canal(ref, self.spacing)

        # ── 4. IM Rod sizing ──────────────────────────────────────────────
        rod_diam, conf = select_rod_diameter(canal_meas["isthmus_diameter_mm"])
        rod_len        = select_rod_length(canal_meas["femur_length_mm"])

        # ── 5. AO classification ──────────────────────────────────────────
        n_frac_levels = self._count_fracture_levels(frac)
        ao_code, ao_desc, pattern = classify_ao(
            len(fragments), frac_angle, n_frac_levels
        )

        return FractureDetectionResult(
            n_fragments             = len(fragments),
            fragments               = fragments,
            ao_code                 = ao_code,
            ao_description          = ao_desc,
            fracture_pattern        = pattern,
            fracture_z_voxel        = frac_z,
            fracture_z_mm           = frac_z_mm,
            fracture_angle_deg      = frac_angle,
            canal_diameter_mm       = canal_meas["isthmus_diameter_mm"],
            canal_measurements      = canal_meas,
            recommended_rod_diameter_mm = rod_diam,
            recommended_rod_length_mm   = rod_len,
            rod_sizing_confidence       = conf,
        )

    def _detect_fracture_plane(
        self,
        frac: np.ndarray,
        fragments: List[BoneFragment],
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the primary fracture plane axial position and tilt angle."""
        if len(fragments) < 2:
            return None, None, 0.0

        # Fracture z ≈ midpoint between proximal and distal main fragments
        prox_cz = fragments[0].centroid_mm[0] / self.spacing[0]
        dist_cz = fragments[1].centroid_mm[0] / self.spacing[0]
        frac_z  = int((prox_cz + dist_cz) / 2.0)
        frac_z_mm = frac_z * self.spacing[0]

        # Estimate tilt from centroid offset in x-y
        dy = fragments[1].centroid_mm[1] - fragments[0].centroid_mm[1]
        dz = abs(fragments[1].centroid_mm[0] - fragments[0].centroid_mm[0])
        angle = float(np.degrees(np.arctan2(abs(dy), dz + 1e-6)))
        return frac_z, frac_z_mm, angle

    def _count_fracture_levels(self, frac: np.ndarray) -> int:
        """Count distinct axial levels where the bone is completely interrupted."""
        axial_counts = frac.sum(axis=(1, 2))
        D = len(axial_counts)
        gap_slices = (axial_counts == 0)
        # Count transitions from bone to gap
        levels = 0
        in_gap = False
        for g in gap_slices:
            if g and not in_gap:
                levels += 1
                in_gap = True
            elif not g:
                in_gap = False
        return max(levels, 1)

    def format_report(self, r: FractureDetectionResult) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║           FRACTURE DETECTION REPORT                         ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            f"  AO/OTA Classification : {r.ao_code} — {r.ao_description}",
            f"  Fracture pattern      : {r.fracture_pattern}",
            f"  Number of fragments   : {r.n_fragments}",
            f"  Fracture angle        : {r.fracture_angle_deg:.1f}°",
        ]
        if r.fracture_z_mm is not None:
            lines.append(f"  Fracture plane (z)    : {r.fracture_z_mm:.1f} mm")
        lines += [
            "",
            "▸ MEDULLARY CANAL",
            f"  Isthmus diameter      : {r.canal_diameter_mm:.1f} mm",
            f"  Femur length          : {r.canal_measurements.get('femur_length_mm', 0):.1f} mm",
            "",
            "▸ IM ROD RECOMMENDATION",
            f"  Rod diameter          : {r.recommended_rod_diameter_mm} mm",
            f"  Rod length            : {r.recommended_rod_length_mm} mm",
            f"  Confidence            : {r.rod_sizing_confidence.upper()}",
            "",
            "▸ FRAGMENTS",
        ]
        for f in r.fragments:
            lines.append(
                f"  [{f.label}]  vol={f.volume_mm3:.0f} mm³  "
                f"centroid=({f.centroid_mm[0]:.1f}, {f.centroid_mm[1]:.1f}, {f.centroid_mm[2]:.1f}) mm"
            )
        lines.append("")
        return "\n".join(lines)