"""
inference/pipeline.py — End-to-end inference pipeline.

Usage
-----
    from inference.pipeline import FemurReconstructionPipeline

    pipe   = FemurReconstructionPipeline.from_checkpoints(
                 seg_ckpt  = "checkpoints/seg_best.pth",
                 comp_ckpt = "checkpoints/comp_best.pth",
             )
    result = pipe.run("path/to/dicom_folder")
    print(result.missing_report.format())
    result.save("results/patient_001")

Output (PipelineResult)
-----------------------
    fractured_mask   : (D, H, W) binary numpy  — segmented fractured bone
    completed_mask   : (D, H, W) binary numpy  — reconstructed intact bone
    missing_mask     : (D, H, W) binary numpy  — missing bone region
    missing_report   : MissingBoneReport
    fracture_result  : FractureDetectionResult
    spacing_mm       : voxel spacing used throughout
    meshes           : dict with 'fractured', 'completed', 'missing' trimesh.Trimesh
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes

from config import cfg
from data.dicom_loader import load_ct_scan
from data.preprocessor import preprocess
from models.unet3d import UNet3D
from models.completion_net import BoneCompletionNet
from analysis.missing_bone_analyzer import MissingBoneAnalyzer, MissingBoneReport
from analysis.fracture_detector import FractureDetector, FractureDetectionResult


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    # Raw masks (D, H, W)
    fractured_mask:   np.ndarray
    completed_mask:   np.ndarray
    missing_mask:     np.ndarray

    # Reports
    missing_report:   MissingBoneReport
    fracture_result:  FractureDetectionResult

    # Geometry
    spacing_mm:       Tuple[float, float, float]
    meshes:           Dict[str, trimesh.Trimesh] = field(default_factory=dict)

    # Timing
    timing:           Dict[str, float] = field(default_factory=dict)

    # ── Convenience ───────────────────────────────────────────────────────
    def summary(self) -> str:
        mr = self.missing_report
        fr = self.fracture_result
        return (
            f"═══════════════════ PIPELINE SUMMARY ═══════════════════\n"
            f"  Fracture type     : {fr.ao_code} — {fr.ao_description}\n"
            f"  Fragments         : {fr.n_fragments}\n"
            f"  Missing volume    : {mr.missing_volume_mm3:.1f} mm³  "
            f"({mr.missing_percent:.1f}% of intact)\n"
            f"  Severity          : {mr.severity.upper()}\n"
            f"  Primary zone      : {mr.primary_fracture_zone}\n"
            f"  Canal diameter    : {fr.canal_diameter_mm:.1f} mm\n"
            f"  IM Rod (diameter) : {fr.recommended_rod_diameter_mm} mm\n"
            f"  IM Rod (length)   : {fr.recommended_rod_length_mm} mm\n"
            f"  Confidence        : {fr.rod_sizing_confidence.upper()}\n"
            f"  Total time        : {sum(self.timing.values()):.1f}s\n"
            f"════════════════════════════════════════════════════════"
        )

    def save(self, output_dir: str | Path):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save masks
        np.save(out / "fractured_mask.npy", self.fractured_mask)
        np.save(out / "completed_mask.npy", self.completed_mask)
        np.save(out / "missing_mask.npy",   self.missing_mask)

        # Save meshes
        for name, mesh in self.meshes.items():
            mesh.export(str(out / f"mesh_{name}.stl"))

        # Save text reports
        with open(out / "missing_bone_report.txt", "w") as f:
            f.write(MissingBoneAnalyzer(self.spacing_mm).format_report(self.missing_report))

        with open(out / "fracture_report.txt", "w") as f:
            from analysis.fracture_detector import FractureDetector
            f.write(FractureDetector(self.spacing_mm).format_report(self.fracture_result))

        with open(out / "summary.txt", "w") as f:
            f.write(self.summary())

        print(f"Results saved to {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 3-D Mesh extraction
# ═══════════════════════════════════════════════════════════════════════════

def mask_to_mesh(
    mask:       np.ndarray,
    spacing_mm: Tuple[float, float, float],
    iso_level:  float = 0.5,
    smooth:     bool  = True,
) -> Optional[trimesh.Trimesh]:
    """
    Convert a binary voxel mask to a watertight trimesh via marching cubes.

    Smoothing uses Laplacian mesh smoothing to remove staircase artefacts
    from voxel boundaries.
    """
    if not mask.any():
        return None

    # Pad to avoid open edges at boundaries
    padded = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0)

    try:
        verts, faces, normals, _ = marching_cubes(padded, level=iso_level, spacing=spacing_mm)
    except ValueError:
        return None

    # Offset verts back by one voxel (accounting for padding)
    verts -= np.array(spacing_mm)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    if smooth:
        trimesh.smoothing.filter_laplacian(mesh, iterations=5)

    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class FemurReconstructionPipeline:
    """
    End-to-end femur reconstruction and missing bone analysis.

    Parameters
    ----------
    seg_model  : trained UNet3D for femur segmentation
    comp_model : trained BoneCompletionNet for bone reconstruction
    device     : torch device
    target_spacing  : resample CT to this spacing (mm) before inference
    target_shape    : fixed volume dimensions fed to the network
    seg_threshold   : sigmoid threshold for segmentation binary mask
    comp_threshold  : sigmoid threshold for completion binary mask
    """

    def __init__(
        self,
        seg_model:       UNet3D,
        comp_model:      BoneCompletionNet,
        device:          torch.device,
        target_spacing:  Tuple[float, float, float] = (1.0, 1.0, 1.5),
        target_shape:    Tuple[int, int, int]        = (128, 128, 64),
        seg_threshold:   float = 0.5,
        comp_threshold:  float = 0.5,
    ):
        self.seg_model       = seg_model.to(device).eval()
        self.comp_model      = comp_model.to(device).eval()
        self.device          = device
        self.target_spacing  = target_spacing
        self.target_shape    = target_shape
        self.seg_threshold   = seg_threshold
        self.comp_threshold  = comp_threshold

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoints(
        cls,
        seg_ckpt:  str | Path,
        comp_ckpt: str | Path,
        device:    str = "cpu",
        **kwargs,
    ) -> "FemurReconstructionPipeline":
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

        # Load segmentation U-Net
        seg_ck  = torch.load(seg_ckpt,  map_location=dev, weights_only=False)
        seg_cfg = seg_ck.get("model_cfg", {})
        seg_model = UNet3D(
            in_channels  = seg_cfg.get("in_channels",  1),
            out_channels = seg_cfg.get("out_channels", 1),
            base_filters = seg_cfg.get("base_filters", cfg.model.base_filters),
            depth        = seg_cfg.get("depth",        cfg.model.seg_depth),
            dropout      = seg_cfg.get("dropout",      0.0),
        )
        seg_model.load_state_dict(seg_ck["model"])

        # Load completion network
        comp_ck = torch.load(comp_ckpt, map_location=dev, weights_only=False)
        comp_model = BoneCompletionNet(
            base_filters = cfg.model.base_filters,
            depth        = 4,
            dropout      = 0.0,
        )
        comp_model.load_state_dict(comp_ck["model"])

        return cls(seg_model, comp_model, dev, **kwargs)

    # ── Main run ──────────────────────────────────────────────────────────

    def run(self, scan_path: str | Path, build_meshes: bool = True) -> PipelineResult:
        """
        Full pipeline:  CT scan path  →  PipelineResult

        Steps
        -----
        1. Load CT scan (DICOM / NIfTI)
        2. Preprocess (HU windowing, resample, crop/pad)
        3. Segment femur (U-Net)
        4. Complete bone (BoneCompletionNet)
        5. Compute missing bone (MissingBoneAnalyzer)
        6. Detect fracture & size IM rod (FractureDetector)
        7. Extract 3-D meshes (marching cubes)
        """
        timing = {}

        # ── 1. Load ───────────────────────────────────────────────────────
        t0 = time.time()
        volume_hu, raw_spacing = load_ct_scan(scan_path)
        timing["load"] = time.time() - t0
        print(f"[1/7] Loaded CT  — shape {volume_hu.shape}  spacing {raw_spacing}  ({timing['load']:.1f}s)")

        # ── 2. Preprocess ─────────────────────────────────────────────────
        t0 = time.time()
        prep = preprocess(
            volume_hu,
            raw_spacing,
            target_spacing = self.target_spacing,
            target_shape   = self.target_shape,
        )
        windowed = prep["windowed"]   # (D, H, W) float32 normalised
        timing["preprocess"] = time.time() - t0
        print(f"[2/7] Preprocessed  — shape {windowed.shape}  ({timing['preprocess']:.1f}s)")

        # ── 3. Segment ────────────────────────────────────────────────────
        t0 = time.time()
        inp_tensor = torch.from_numpy(windowed[np.newaxis, np.newaxis]).to(self.device)
        with torch.no_grad():
            seg_logits  = self.seg_model(inp_tensor)
            seg_probs   = torch.sigmoid(seg_logits)
            frac_mask_t = (seg_probs > self.seg_threshold).float()
        fractured_mask = frac_mask_t[0, 0].cpu().numpy().astype(bool)
        timing["segment"] = time.time() - t0
        print(f"[3/7] Segmented  — bone voxels: {fractured_mask.sum():,}  ({timing['segment']:.1f}s)")

        # ── 4. Complete ───────────────────────────────────────────────────
        t0 = time.time()
        frac_t = frac_mask_t   # (1, 1, D, H, W) already on device
        with torch.no_grad():
            comp_logits  = self.comp_model(frac_t)
            comp_probs   = torch.sigmoid(comp_logits)
            comp_mask_t  = (comp_probs > self.comp_threshold).float()
        completed_mask = comp_mask_t[0, 0].cpu().numpy().astype(bool)
        timing["complete"] = time.time() - t0
        print(f"[4/7] Reconstructed  — intact voxels: {completed_mask.sum():,}  ({timing['complete']:.1f}s)")

        # ── 5. Missing bone ───────────────────────────────────────────────
        t0 = time.time()
        analyzer       = MissingBoneAnalyzer(spacing_mm=self.target_spacing)
        missing_report = analyzer.analyze(fractured_mask, completed_mask)
        missing_mask   = missing_report.missing_mask
        timing["analyze"] = time.time() - t0
        print(
            f"[5/7] Missing bone  — "
            f"{missing_report.missing_volume_mm3:.0f} mm³  "
            f"({missing_report.missing_percent:.1f}%)  "
            f"({timing['analyze']:.1f}s)"
        )

        # ── 6. Fracture detection & IM rod sizing ─────────────────────────
        t0 = time.time()
        detector      = FractureDetector(spacing_mm=self.target_spacing)
        frac_result   = detector.detect(fractured_mask, completed_mask)
        timing["detect"] = time.time() - t0
        print(
            f"[6/7] Fracture  — {frac_result.ao_code}  "
            f"fragments: {frac_result.n_fragments}  "
            f"rod: {frac_result.recommended_rod_diameter_mm}×{frac_result.recommended_rod_length_mm} mm  "
            f"({timing['detect']:.1f}s)"
        )

        # ── 7. Meshes ─────────────────────────────────────────────────────
        meshes = {}
        if build_meshes:
            t0 = time.time()
            for name, mask in [
                ("fractured", fractured_mask),
                ("completed", completed_mask),
                ("missing",   missing_mask),
            ]:
                m = mask_to_mesh(mask, self.target_spacing)
                if m is not None:
                    meshes[name] = m
            timing["mesh"] = time.time() - t0
            print(f"[7/7] Meshes built  ({timing['mesh']:.1f}s)")

        result = PipelineResult(
            fractured_mask  = fractured_mask,
            completed_mask  = completed_mask,
            missing_mask    = missing_mask,
            missing_report  = missing_report,
            fracture_result = frac_result,
            spacing_mm      = self.target_spacing,
            meshes          = meshes,
            timing          = timing,
        )

        print("\n" + result.summary())
        return result