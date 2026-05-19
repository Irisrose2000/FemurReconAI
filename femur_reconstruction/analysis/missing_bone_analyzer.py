"""
analysis/missing_bone_analyzer.py — Missing Bone Quantification Engine.

Given:
    fractured_mask  : (D, H, W) binary — what's physically present
    completed_mask  : (D, H, W) binary — what the intact bone should look like

Computes:
    ┌─────────────────────────────────────────────────────────────┐
    │  missing_mask        — voxels that are in complete but not  │
    │                        in fractured (the actual bone loss)  │
    │  missing_volume_mm3  — physical volume of missing bone (mm³)│
    │  missing_percent     — % of intact bone that is absent      │
    │  gap_regions         — per-gap: location, size, shape desc  │
    │  anatomical_zone     — which femur region each gap is in    │
    │  centroid_mm         — 3-D centroid of the missing region   │
    │  axial_profile       — missing voxel count per axial slice  │
    └─────────────────────────────────────────────────────────────┘
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
class GapRegion:
    """Describes one contiguous missing-bone region."""
    id:                  int
    volume_mm3:          float
    volume_percent:      float          # % of total intact volume
    centroid_voxels:     Tuple[float, float, float]
    centroid_mm:         Tuple[float, float, float]
    bounding_box:        Tuple          # (z0,y0,x0,z1,y1,x1)
    axial_extent_mm:     float          # gap height along femur axis
    max_cross_section_mm2: float        # largest cross-section area
    anatomical_zone:     str            # head/neck/trochanteric/diaphysis/distal
    shape_descriptor:    str            # transverse / wedge / comminuted


@dataclass
class MissingBoneReport:
    """Full output of MissingBoneAnalyzer.analyze()."""
    # ── Volumes ────────────────────────────────────────────────────────────
    intact_volume_mm3:    float
    fractured_volume_mm3: float
    missing_volume_mm3:   float
    missing_percent:      float         # missing / intact × 100

    # ── Spatial ───────────────────────────────────────────────────────────
    missing_mask:         np.ndarray    # (D, H, W) bool
    centroid_mm:          Optional[Tuple[float, float, float]]

    # ── Gap details ───────────────────────────────────────────────────────
    gap_regions:          List[GapRegion] = field(default_factory=list)
    n_gaps:               int = 0

    # ── Axial profile ─────────────────────────────────────────────────────
    axial_profile:        Optional[np.ndarray] = None   # shape (D,)
    axial_profile_mm3:    Optional[np.ndarray] = None

    # ── Fracture summary ─────────────────────────────────────────────────
    primary_fracture_zone: str = "unknown"
    total_axial_gap_mm:   float = 0.0
    severity:             str = "minor"   # minor / moderate / severe / critical


# ═══════════════════════════════════════════════════════════════════════════
# Anatomical zone classification
# ═══════════════════════════════════════════════════════════════════════════

FEMUR_ZONES = [
    (0.00, 0.08, "head"),
    (0.08, 0.18, "neck"),
    (0.18, 0.28, "trochanteric"),
    (0.28, 0.75, "diaphysis"),
    (0.75, 1.00, "distal"),
]

def classify_zone(z_norm: float) -> str:
    """Map a normalised axial position [0,1] to an anatomical zone name."""
    for lo, hi, name in FEMUR_ZONES:
        if lo <= z_norm <= hi:
            return name
    return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Main Analyzer
# ═══════════════════════════════════════════════════════════════════════════

class MissingBoneAnalyzer:
    """
    Parameters
    ----------
    spacing_mm : (sz, sy, sx) — voxel spacing in mm (from the CT loader)
    min_gap_volume_mm3 : ignore missing regions smaller than this (noise filter)
    """

    def __init__(
        self,
        spacing_mm:          Tuple[float, float, float] = (1.0, 1.0, 1.0),
        min_gap_volume_mm3:  float = 50.0,
    ):
        self.spacing   = np.array(spacing_mm, dtype=np.float64)
        self.voxel_vol = float(np.prod(self.spacing))   # mm³ per voxel
        self.voxel_area_xy = float(self.spacing[1] * self.spacing[2])  # mm²
        self.min_gap_vol = min_gap_volume_mm3

    # ── Public API ─────────────────────────────────────────────────────────

    def analyze(
        self,
        fractured_mask: np.ndarray,
        completed_mask: np.ndarray,
    ) -> MissingBoneReport:
        """
        Parameters
        ----------
        fractured_mask : (D, H, W) float or bool — segmented fractured bone
        completed_mask : (D, H, W) float or bool — reconstructed intact bone

        Returns
        -------
        MissingBoneReport
        """
        frac = (fractured_mask > 0.5).astype(bool)
        comp = (completed_mask > 0.5).astype(bool)

        # ── 1. Missing voxels ─────────────────────────────────────────────
        # Missing = in complete but NOT in fractured
        missing = comp & ~frac

        # ── 2. Volume calculations ────────────────────────────────────────
        intact_vol   = float(comp.sum()    * self.voxel_vol)
        frac_vol     = float(frac.sum()    * self.voxel_vol)
        missing_vol  = float(missing.sum() * self.voxel_vol)
        missing_pct  = (missing_vol / intact_vol * 100.0) if intact_vol > 0 else 0.0

        # ── 3. Centroid of missing region ─────────────────────────────────
        centroid_mm = self._centroid_mm(missing) if missing.any() else None

        # ── 4. Axial profile (per-slice missing count) ────────────────────
        axial_profile     = missing.sum(axis=(1, 2)).astype(np.float32)
        axial_profile_mm3 = axial_profile * self.voxel_vol

        # ── 5. Connected component analysis of missing regions ────────────
        gap_regions = self._analyze_gap_regions(missing, comp)

        # ── 6. Severity ───────────────────────────────────────────────────
        severity = self._classify_severity(missing_pct, len(gap_regions))

        # ── 7. Primary fracture zone ──────────────────────────────────────
        if gap_regions:
            # Zone of the largest gap
            primary = max(gap_regions, key=lambda g: g.volume_mm3)
            primary_zone = primary.anatomical_zone
            total_axial_gap = sum(g.axial_extent_mm for g in gap_regions)
        else:
            primary_zone   = classify_zone(0.5)
            total_axial_gap = 0.0

        return MissingBoneReport(
            intact_volume_mm3     = intact_vol,
            fractured_volume_mm3  = frac_vol,
            missing_volume_mm3    = missing_vol,
            missing_percent       = missing_pct,
            missing_mask          = missing,
            centroid_mm           = centroid_mm,
            gap_regions           = gap_regions,
            n_gaps                = len(gap_regions),
            axial_profile         = axial_profile,
            axial_profile_mm3     = axial_profile_mm3,
            primary_fracture_zone = primary_zone,
            total_axial_gap_mm    = total_axial_gap,
            severity              = severity,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _centroid_mm(self, mask: np.ndarray) -> Tuple[float, float, float]:
        coords = np.argwhere(mask).astype(np.float64)
        c = coords.mean(axis=0)          # (z, y, x) in voxels
        return tuple(float(c[i] * self.spacing[i]) for i in range(3))

    def _analyze_gap_regions(
        self,
        missing: np.ndarray,
        complete: np.ndarray,
    ) -> List[GapRegion]:
        """Label connected components of the missing mask and measure each."""
        if not missing.any():
            return []

        labeled, n = cc_label(missing, return_num=True)
        props       = regionprops(labeled)
        D           = missing.shape[0]
        regions     = []

        for prop in props:
            vol_mm3 = prop.area * self.voxel_vol
            if vol_mm3 < self.min_gap_vol:
                continue  # too small — noise

            cz, cy, cx = prop.centroid
            intact_vol = float(complete.sum() * self.voxel_vol)

            z0, y0, x0, z1, y1, x1 = prop.bbox

            # Axial extent
            axial_mm = (z1 - z0) * float(self.spacing[0])

            # Maximum cross-sectional area (in any axial slice)
            component_mask = labeled == prop.label
            cross_areas = component_mask.sum(axis=(1, 2)) * self.voxel_area_xy
            max_cross   = float(cross_areas.max())

            # Anatomical zone of centroid
            z_norm = cz / D
            zone   = classify_zone(z_norm)

            # Shape descriptor
            shape = self._describe_shape(
                component_mask, axial_mm, max_cross, z0, z1
            )

            regions.append(GapRegion(
                id                    = prop.label,
                volume_mm3            = vol_mm3,
                volume_percent        = vol_mm3 / intact_vol * 100.0 if intact_vol > 0 else 0.0,
                centroid_voxels       = (cz, cy, cx),
                centroid_mm           = (
                    cz * self.spacing[0],
                    cy * self.spacing[1],
                    cx * self.spacing[2],
                ),
                bounding_box          = (z0, y0, x0, z1, y1, x1),
                axial_extent_mm       = axial_mm,
                max_cross_section_mm2 = max_cross,
                anatomical_zone       = zone,
                shape_descriptor      = shape,
            ))

        # Sort by volume (largest first)
        regions.sort(key=lambda g: g.volume_mm3, reverse=True)
        # Re-index
        for i, g in enumerate(regions, start=1):
            g.id = i
        return regions

    def _describe_shape(
        self,
        mask: np.ndarray,
        axial_mm: float,
        max_cross_mm2: float,
        z0: int, z1: int,
    ) -> str:
        """Classify gap shape from geometry."""
        if axial_mm < 5.0:
            return "transverse"

        # Check if cross-section varies greatly along the axis → wedge
        slices = mask[z0:z1].sum(axis=(1, 2))
        if slices.max() > 0:
            ratio = slices.min() / slices.max()
            if ratio < 0.3:
                return "wedge"

        # Multiple sub-peaks → comminuted
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(slices, height=slices.max() * 0.2)
        if len(peaks) >= 2:
            return "comminuted"

        if axial_mm > 20.0:
            return "segmental"

        return "oblique"

    def _classify_severity(self, missing_pct: float, n_gaps: int) -> str:
        if missing_pct < 5.0 and n_gaps <= 1:
            return "minor"
        elif missing_pct < 15.0 and n_gaps <= 2:
            return "moderate"
        elif missing_pct < 30.0 or n_gaps <= 4:
            return "severe"
        else:
            return "critical"

    # ── Report formatting ──────────────────────────────────────────────────

    def format_report(self, report: MissingBoneReport) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════════════╗",
            "║           MISSING BONE ANALYSIS REPORT                      ║",
            "╚══════════════════════════════════════════════════════════════╝",
            "",
            "▸ VOLUME SUMMARY",
            f"  Intact femur volume    : {report.intact_volume_mm3:>10.1f} mm³",
            f"  Fractured bone volume  : {report.fractured_volume_mm3:>10.1f} mm³",
            f"  Missing bone volume    : {report.missing_volume_mm3:>10.1f} mm³",
            f"  Missing percentage     : {report.missing_percent:>10.2f} %",
            f"  Severity               : {report.severity.upper()}",
            "",
            "▸ FRACTURE DETAILS",
            f"  Primary zone           : {report.primary_fracture_zone}",
            f"  Number of gaps         : {report.n_gaps}",
            f"  Total axial gap        : {report.total_axial_gap_mm:.1f} mm",
        ]

        if report.centroid_mm:
            c = report.centroid_mm
            lines.append(
                f"  Missing region centroid: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) mm"
            )

        if report.gap_regions:
            lines += ["", "▸ GAP BREAKDOWN"]
            for g in report.gap_regions:
                lines += [
                    f"  Gap #{g.id}",
                    f"    Volume         : {g.volume_mm3:.1f} mm³  ({g.volume_percent:.1f}% of intact)",
                    f"    Zone           : {g.anatomical_zone}",
                    f"    Shape          : {g.shape_descriptor}",
                    f"    Axial extent   : {g.axial_extent_mm:.1f} mm",
                    f"    Max cross-sec  : {g.max_cross_section_mm2:.1f} mm²",
                    f"    Centroid (mm)  : ({g.centroid_mm[0]:.1f}, {g.centroid_mm[1]:.1f}, {g.centroid_mm[2]:.1f})",
                ]

        lines.append("")
        return "\n".join(lines)