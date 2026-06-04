"""
inference/pipeline.py — End-to-end inference pipeline.
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
    fractured_mask: np.ndarray
    completed_mask: np.ndarray
    missing_mask: np.ndarray

    missing_report: MissingBoneReport
    fracture_result: FractureDetectionResult

    spacing_mm: Tuple[float, float, float]
    meshes: Dict[str, trimesh.Trimesh] = field(default_factory=dict)

    timing: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        mr = self.missing_report
        fr = self.fracture_result

        return (
            f"═══════════════════ PIPELINE SUMMARY ═══════════════════\n"
            f"  Fracture type     : {fr.ao_code} — {fr.ao_description}\n"
            f"  Fragments         : {fr.n_fragments}\n"
            f"  Missing volume    : {mr.missing_volume_mm3:.1f} mm³ "
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

        np.save(out / "fractured_mask.npy", self.fractured_mask)
        np.save(out / "completed_mask.npy", self.completed_mask)
        np.save(out / "missing_mask.npy", self.missing_mask)

        for name, mesh in self.meshes.items():
            mesh.export(str(out / f"mesh_{name}.stl"))

        with open(out / "summary.txt", "w", encoding="utf-8") as f:
            f.write(self.summary())

        print(f"Results saved to {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Mesh extraction
# ═══════════════════════════════════════════════════════════════════════════

def mask_to_mesh(
    mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    iso_level: float = 0.5,
    smooth: bool = True,
) -> Optional[trimesh.Trimesh]:

    if not mask.any():
        return None

    padded = np.pad(mask.astype(np.float32), 1, mode="constant")

    try:
        verts, faces, normals, _ = marching_cubes(
            padded,
            level=iso_level,
            spacing=spacing_mm
        )
    except ValueError:
        return None

    verts -= np.array(spacing_mm)

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=True
    )

    if smooth:
        trimesh.smoothing.filter_laplacian(mesh, iterations=5)

    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

class FemurReconstructionPipeline:

    def __init__(
        self,
        seg_model: UNet3D,
        comp_model: BoneCompletionNet,
        device: torch.device,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.5),
        target_shape: Tuple[int, int, int] = (128, 128, 64),
        seg_threshold: float = 0.5,
        comp_threshold: float = 0.5,
    ):
        self.seg_model = seg_model.to(device).eval()
        self.comp_model = comp_model.to(device).eval()

        self.device = device
        self.target_spacing = target_spacing
        self.target_shape = target_shape

        self.seg_threshold = seg_threshold
        self.comp_threshold = comp_threshold

    # ──────────────────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoints(
        cls,
        seg_ckpt: str | Path,
        comp_ckpt: str | Path,
        device: str = "cpu",
        **kwargs,
    ):

        dev = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )

        # Segmentation model
        seg_ck = torch.load(seg_ckpt, map_location=dev, weights_only=False)

        seg_cfg = seg_ck.get("model_cfg", {})

        seg_model = UNet3D(
            in_channels=seg_cfg.get("in_channels", 1),
            out_channels=seg_cfg.get("out_channels", 1),
            base_filters=seg_cfg.get("base_filters", cfg.model.base_filters),
            depth=seg_cfg.get("depth", cfg.model.seg_depth),
            dropout=seg_cfg.get("dropout", 0.0),
        )

        seg_model.load_state_dict(seg_ck["model"])

        # Completion model
        comp_ck = torch.load(comp_ckpt, map_location=dev, weights_only=False)

        comp_model = BoneCompletionNet(
            base_filters=cfg.model.base_filters,
            depth=4,
            dropout=0.0,
        )

        comp_model.load_state_dict(comp_ck["model"])

        return cls(seg_model, comp_model, dev, **kwargs)

    # ──────────────────────────────────────────────────────────────────────
    # Main pipeline
    # ──────────────────────────────────────────────────────────────────────

    def run(self, scan_path: str | Path, build_meshes: bool = True):

        timing = {}

        # ── 1. Load ───────────────────────────────────────────────────────

        t0 = time.time()

        scan_path = str(scan_path)

        if scan_path.endswith(".npz"):

            data = np.load(scan_path)

            # synthetic fractured volume
            volume_hu = data["fractured"].astype(np.float32)

            # synthetic spacing
            raw_spacing = (1.0, 1.0, 1.0)

        else:

            volume_hu, raw_spacing = load_ct_scan(scan_path)

        timing["load"] = time.time() - t0

        print(
            f"[1/7] Loaded scan — shape {volume_hu.shape} "
            f"spacing {raw_spacing} ({timing['load']:.1f}s)"
        )

        # ── 2. Preprocess ────────────────────────────────────────────────

        t0 = time.time()

        prep = preprocess(
            volume_hu,
            raw_spacing,
            target_spacing=self.target_spacing,
            target_shape=self.target_shape,
        )

        windowed = prep["windowed"]

        timing["preprocess"] = time.time() - t0

        print(
            f"[2/7] Preprocessed — shape {windowed.shape} "
            f"({timing['preprocess']:.1f}s)"
        )

        # ── 3. Segmentation ──────────────────────────────────────────────

        t0 = time.time()

        inp_tensor = torch.from_numpy(
            windowed[np.newaxis, np.newaxis]
        ).to(self.device)

        with torch.no_grad():

            seg_logits = self.seg_model(inp_tensor)

            seg_probs = torch.sigmoid(seg_logits)

            frac_mask_t = (seg_probs > self.seg_threshold).float()

        fractured_mask = frac_mask_t[0, 0].cpu().numpy().astype(bool)

        timing["segment"] = time.time() - t0

        print(
            f"[3/7] Segmented — voxels: {fractured_mask.sum():,} "
            f"({timing['segment']:.1f}s)"
        )

        # ── 4. Completion ────────────────────────────────────────────────

        t0 = time.time()

        with torch.no_grad():

            comp_logits = self.comp_model(frac_mask_t)

            comp_probs = torch.sigmoid(comp_logits)

            comp_mask_t = (comp_probs > self.comp_threshold).float()

        completed_mask = comp_mask_t[0, 0].cpu().numpy().astype(bool)

        timing["complete"] = time.time() - t0

        print(
            f"[4/7] Completed — voxels: {completed_mask.sum():,} "
            f"({timing['complete']:.1f}s)"
        )

        # ── 5. Missing bone analysis ─────────────────────────────────────

        t0 = time.time()

        analyzer = MissingBoneAnalyzer(spacing_mm=self.target_spacing)

        missing_report = analyzer.analyze(
            fractured_mask,
            completed_mask
        )

        missing_mask = missing_report.missing_mask

        timing["analyze"] = time.time() - t0

        print(
            f"[5/7] Missing bone — "
            f"{missing_report.missing_volume_mm3:.1f} mm³ "
            f"({missing_report.missing_percent:.1f}%)"
        )

        # ── 6. Fracture detection ────────────────────────────────────────

        t0 = time.time()

        detector = FractureDetector(spacing_mm=self.target_spacing)

        frac_result = detector.detect(
            fractured_mask,
            completed_mask
        )

        timing["detect"] = time.time() - t0

        print(
            f"[6/7] Fracture — {frac_result.ao_code} "
            f"rod {frac_result.recommended_rod_diameter_mm} × "
            f"{frac_result.recommended_rod_length_mm} mm"
        )

        # ── 7. Mesh generation ───────────────────────────────────────────

        meshes = {}

        if build_meshes:

            t0 = time.time()

            for name, mask in [
                ("fractured", fractured_mask),
                ("completed", completed_mask),
                ("missing", missing_mask),
            ]:

                mesh = mask_to_mesh(mask, self.target_spacing)

                if mesh is not None:
                    meshes[name] = mesh

            timing["mesh"] = time.time() - t0

            print(f"[7/7] Meshes built ({timing['mesh']:.1f}s)")

        # ── Result ───────────────────────────────────────────────────────

        result = PipelineResult(
            fractured_mask=fractured_mask,
            completed_mask=completed_mask,
            missing_mask=missing_mask,
            missing_report=missing_report,
            fracture_result=frac_result,
            spacing_mm=self.target_spacing,
            meshes=meshes,
            timing=timing,
        )

        print("\n" + result.summary())

        return result