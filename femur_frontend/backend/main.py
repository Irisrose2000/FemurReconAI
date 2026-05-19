"""
backend/main.py — FastAPI backend for FemurRecon AI

Endpoints
---------
POST /api/analyse          Upload CT scan → run full ML pipeline → return job ID
GET  /api/status/{job_id}  Poll job status + progress
GET  /api/results/{job_id} Full analysis results (JSON)
GET  /api/mesh/{job_id}/{type}  Stream STL mesh (fractured | completed | missing)
GET  /api/health           Health check
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import numpy as np

app = FastAPI(title="FemurRecon AI", version="1.0.0")

# ── CORS — allow local React dev server ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store (use Redis / DB in production) ────────────────────
JOBS: Dict[str, dict] = {}
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Background task
# ═══════════════════════════════════════════════════════════════════════════

def _run_pipeline(job_id: str, scan_path: str, file_type: str):
    """Run the full ML pipeline in a background thread."""
    try:
        JOBS[job_id]["status"]   = "running"
        JOBS[job_id]["progress"] = 5
        JOBS[job_id]["step"]     = "Loading CT scan…"

        # ── Try to load real pipeline ─────────────────────────────────────
        pipeline = None
        seg_ckpt  = "checkpoints/seg_best.pth"
        comp_ckpt = "checkpoints/comp_best.pth"

        if Path(seg_ckpt).exists() and Path(comp_ckpt).exists():
            from inference.pipeline import FemurReconstructionPipeline
            pipeline = FemurReconstructionPipeline.from_checkpoints(
                seg_ckpt=seg_ckpt, comp_ckpt=comp_ckpt, device="cpu"
            )

        JOBS[job_id]["progress"] = 15
        JOBS[job_id]["step"]     = "Preprocessing volume…"
        time.sleep(0.5)

        if pipeline is not None:
            JOBS[job_id]["progress"] = 30
            JOBS[job_id]["step"]     = "Segmenting femur (U-Net)…"
            result = pipeline.run(scan_path, build_meshes=True)
        else:
            # Demo mode
            result = _demo_result()

        JOBS[job_id]["progress"] = 80
        JOBS[job_id]["step"]     = "Extracting 3D meshes…"
        time.sleep(0.3)

        # ── Serialise result ──────────────────────────────────────────────
        job_dir = RESULTS_DIR / job_id
        job_dir.mkdir(exist_ok=True)

        # Save meshes as STL
        for name, mesh in result.meshes.items():
            mesh.export(str(job_dir / f"mesh_{name}.stl"))

        # Save numpy masks
        np.save(str(job_dir / "fractured_mask.npy"), result.fractured_mask)
        np.save(str(job_dir / "completed_mask.npy"),  result.completed_mask)
        np.save(str(job_dir / "missing_mask.npy"),    result.missing_mask)

        # Build JSON payload
        mr = result.missing_report
        fr = result.fracture_result

        payload = {
            "job_id": job_id,
            "volumes": {
                "intact_mm3":    round(mr.intact_volume_mm3, 1),
                "fractured_mm3": round(mr.fractured_volume_mm3, 1),
                "missing_mm3":   round(mr.missing_volume_mm3, 1),
                "missing_pct":   round(mr.missing_percent, 2),
            },
            "severity":     mr.severity,
            "fracture_zone": mr.primary_fracture_zone,
            "n_gaps":       mr.n_gaps,
            "total_gap_mm": round(mr.total_axial_gap_mm, 1),
            "axial_profile": {
                "voxels": mr.axial_profile.tolist() if mr.axial_profile is not None else [],
                "mm3":    mr.axial_profile_mm3.tolist() if mr.axial_profile_mm3 is not None else [],
            },
            "gap_regions": [
                {
                    "id":             g.id,
                    "volume_mm3":     round(g.volume_mm3, 1),
                    "volume_pct":     round(g.volume_percent, 2),
                    "zone":           g.anatomical_zone,
                    "shape":          g.shape_descriptor,
                    "axial_mm":       round(g.axial_extent_mm, 1),
                    "max_cross_mm2":  round(g.max_cross_section_mm2, 1),
                    "centroid_mm":    [round(x, 1) for x in g.centroid_mm],
                }
                for g in mr.gap_regions
            ],
            "fracture": {
                "ao_code":       fr.ao_code,
                "ao_description": fr.ao_description,
                "pattern":       fr.fracture_pattern,
                "n_fragments":   fr.n_fragments,
                "angle_deg":     round(fr.fracture_angle_deg, 1),
                "fracture_z_mm": round(fr.fracture_z_mm, 1) if fr.fracture_z_mm else None,
            },
            "canal": {
                "isthmus_mm":   round(fr.canal_diameter_mm, 1),
                "mean_mm":      round(fr.canal_measurements.get("mean_canal_mm", 0), 1),
                "femur_len_mm": round(fr.canal_measurements.get("femur_length_mm", 0), 1),
            },
            "rod": {
                "diameter_mm":  fr.recommended_rod_diameter_mm,
                "length_mm":    fr.recommended_rod_length_mm,
                "confidence":   fr.rod_sizing_confidence,
            },
            "fragments": [
                {
                    "id":          f.id,
                    "label":       f.label,
                    "volume_mm3":  round(f.volume_mm3, 1),
                    "centroid_mm": [round(x, 1) for x in f.centroid_mm],
                    "is_main":     f.is_main,
                }
                for f in fr.fragments
            ],
            "meshes_available": list(result.meshes.keys()),
            "timing": result.timing,
            "spacing_mm": list(result.spacing_mm),
        }

        with open(job_dir / "result.json", "w") as f:
            json.dump(payload, f, indent=2)

        JOBS[job_id].update({
            "status":   "complete",
            "progress": 100,
            "step":     "Done",
            "result":   payload,
        })

    except Exception as e:
        JOBS[job_id].update({
            "status":  "error",
            "error":   str(e),
            "trace":   traceback.format_exc(),
            "step":    "Error",
        })


def _demo_result():
    """Return a plausible demo PipelineResult."""
    from analysis.missing_bone_analyzer import MissingBoneReport, GapRegion
    from analysis.fracture_detector import FractureDetectionResult, BoneFragment
    from inference.pipeline import mask_to_mesh, PipelineResult

    D, H, W = 64, 64, 32
    sp = (1.5, 1.0, 1.0)
    zz, yy, xx = np.mgrid[:D, :H, :W]
    cy, cx = H // 2, W // 2
    comp = ((yy - cy) ** 2 + (xx - cx) ** 2 < 12 ** 2).astype(bool)
    frac = comp.copy()
    frac[25:32] = False
    missing = comp & ~frac
    axial = missing.sum(axis=(1, 2)).astype(np.float32) * float(np.prod(sp))

    gap = GapRegion(
        id=1, volume_mm3=3200.0, volume_percent=12.5,
        centroid_voxels=(28.0, 32.0, 16.0),
        centroid_mm=(42.0, 32.0, 16.0),
        bounding_box=(25, 20, 10, 32, 44, 22),
        axial_extent_mm=10.5, max_cross_section_mm2=120.0,
        anatomical_zone="diaphysis", shape_descriptor="transverse",
    )
    mr = MissingBoneReport(
        intact_volume_mm3=25600.0, fractured_volume_mm3=22400.0,
        missing_volume_mm3=3200.0, missing_percent=12.5,
        missing_mask=missing, centroid_mm=(42.0, 32.0, 16.0),
        gap_regions=[gap], n_gaps=1,
        axial_profile=missing.sum(axis=(1, 2)).astype(np.float32),
        axial_profile_mm3=axial,
        primary_fracture_zone="diaphysis", total_axial_gap_mm=10.5, severity="moderate",
    )
    frags = [
        BoneFragment(1, 12000.0, (18.0, 32.0, 16.0), (0,10,5,25,54,27), True, "proximal"),
        BoneFragment(2, 10400.0, (54.0, 32.0, 16.0), (32,10,5,64,54,27), True, "distal"),
    ]
    fr = FractureDetectionResult(
        n_fragments=2, fragments=frags,
        ao_code="A1", ao_description="Simple transverse fracture",
        fracture_pattern="transverse", fracture_z_voxel=28, fracture_z_mm=42.0,
        fracture_angle_deg=3.5, canal_diameter_mm=11.0,
        canal_measurements={"isthmus_diameter_mm": 11.0, "mean_canal_mm": 12.3, "femur_length_mm": 420.0},
        recommended_rod_diameter_mm=10.0, recommended_rod_length_mm=400.0,
        rod_sizing_confidence="high",
    )
    meshes = {k: v for k, v in {
        "fractured": mask_to_mesh(frac, sp),
        "completed": mask_to_mesh(comp, sp),
        "missing":   mask_to_mesh(missing, sp),
    }.items() if v is not None}

    return PipelineResult(
        fractured_mask=frac, completed_mask=comp, missing_mask=missing,
        missing_report=mr, fracture_result=fr, spacing_mm=sp,
        meshes=meshes, timing={"demo": 0.1},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/api/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/analyse")
async def analyse(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    job_id  = str(uuid.uuid4())[:8]
    tmpdir  = tempfile.mkdtemp()
    content = await file.read()

    # Detect file type and extract
    fname   = file.filename or ""
    if fname.endswith(".zip"):
        zip_path = Path(tmpdir) / "scan.zip"
        zip_path.write_bytes(content)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)
        scan_path = tmpdir
        file_type = "dicom"
    elif fname.endswith(".nii.gz"):
        p = Path(tmpdir) / "scan.nii.gz"
        p.write_bytes(content)
        scan_path = str(p)
        file_type = "nifti"
    elif fname.endswith(".nii"):
        p = Path(tmpdir) / "scan.nii"
        p.write_bytes(content)
        scan_path = str(p)
        file_type = "nifti"
    else:
        # Treat as demo
        scan_path = ""
        file_type = "demo"

    JOBS[job_id] = {
        "status":   "queued",
        "progress": 0,
        "step":     "Queued",
        "result":   None,
        "error":    None,
    }

    background_tasks.add_task(_run_pipeline, job_id, scan_path, file_type)
    return {"job_id": job_id}


@app.post("/api/analyse/demo")
async def analyse_demo(background_tasks: BackgroundTasks):
    """Run demo analysis without uploading a file."""
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {"status": "queued", "progress": 0, "step": "Queued", "result": None, "error": None}
    background_tasks.add_task(_run_pipeline, job_id, "", "demo")
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    j = JOBS[job_id]
    return {
        "job_id":   job_id,
        "status":   j["status"],
        "progress": j["progress"],
        "step":     j["step"],
        "error":    j.get("error"),
    }


@app.get("/api/results/{job_id}")
def results(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    j = JOBS[job_id]
    if j["status"] != "complete":
        raise HTTPException(400, f"Job is {j['status']}")
    return j["result"]


@app.get("/api/mesh/{job_id}/{mesh_type}")
def get_mesh(job_id: str, mesh_type: str):
    if mesh_type not in ("fractured", "completed", "missing"):
        raise HTTPException(400, "mesh_type must be fractured | completed | missing")
    stl_path = RESULTS_DIR / job_id / f"mesh_{mesh_type}.stl"
    if not stl_path.exists():
        raise HTTPException(404, "Mesh not found")
    return StreamingResponse(
        io.BytesIO(stl_path.read_bytes()),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="femur_{mesh_type}.stl"'},
    )