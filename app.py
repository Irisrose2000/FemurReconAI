"""
app.py — Streamlit web application for femur fracture analysis.

Run:
    streamlit run app.py

Tabs:
    1. Upload & Analyse   — Upload CT scan, run full pipeline
    2. 3D Viewer          — Interactive 3D bone reconstruction
    3. Missing Bone       — Volume stats + axial profile
    4. IM Rod Sizing      — Measurement table + rod recommendation
    5. Report             — Full text report (downloadable)
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import plotly.graph_objects as go

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="FemurRecon AI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# Imports (deferred so page loads fast)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading AI models…")
def load_pipeline(seg_ckpt: str, comp_ckpt: str, device: str = "cpu"):
    from inference.pipeline import FemurReconstructionPipeline
    return FemurReconstructionPipeline.from_checkpoints(
        seg_ckpt=seg_ckpt, comp_ckpt=comp_ckpt, device=device
    )


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bone.png", width=64)
    st.title("FemurRecon AI")
    st.markdown("**AI-powered femur fracture analysis & IM rod sizing**")
    st.divider()

    st.subheader("⚙️ Model Settings")
    seg_ckpt_path  = st.text_input("Segmentation checkpoint", value="checkpoints/seg_best.pth")
    comp_ckpt_path = st.text_input("Completion checkpoint",   value="checkpoints/comp_best.pth")
    device_choice  = st.radio("Device", ["cpu", "cuda"], index=0)
    load_model_btn = st.button("🔄 Load / Reload Models", type="primary")

    st.divider()
    st.subheader("🔧 Inference Settings")
    seg_thr  = st.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
    comp_thr = st.slider("Completion threshold",   0.1, 0.9, 0.5, 0.05)
    show_rod = st.toggle("Show IM Rod in 3D view", value=True)
    show_ghost = st.toggle("Show ghost intact bone", value=True)

    st.divider()
    st.caption("⚠️ For research use only. Not a medical device.")


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

pipeline = None

ckpt_available = (
    Path(seg_ckpt_path).exists() and Path(comp_ckpt_path).exists()
)

if load_model_btn or ckpt_available:
    if ckpt_available:
        try:
            pipeline = load_pipeline(seg_ckpt_path, comp_ckpt_path, device_choice)
            st.sidebar.success("✅ Models loaded")
        except Exception as e:
            st.sidebar.error(f"Model load failed: {e}")
    else:
        st.sidebar.warning("Checkpoint files not found — running in demo mode")


# ═══════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════

tab_upload, tab_3d, tab_missing, tab_rod, tab_report = st.tabs([
    "📤 Upload & Analyse",
    "🧊 3D Viewer",
    "🔴 Missing Bone",
    "📏 IM Rod Sizing",
    "📄 Report",
])


# ── Helper: demo result from random data ─────────────────────────────────

def _make_demo_result():
    """Generate plausible demo output without real CT data."""
    from analysis.missing_bone_analyzer import MissingBoneReport, GapRegion
    from analysis.fracture_detector import FractureDetectionResult, BoneFragment
    import trimesh

    D, H, W = 64, 64, 32
    sp = (1.5, 1.0, 1.0)

    # Synthetic cylinder (femur shaft)
    zz, yy, xx = np.mgrid[:D, :H, :W]
    cy, cx = H // 2, W // 2
    comp = ((yy - cy) ** 2 + (xx - cx) ** 2 < 12 ** 2).astype(bool)
    frac = comp.copy()
    frac[25:32] = False   # gap

    missing = comp & ~frac
    axial   = missing.sum(axis=(1, 2)).astype(np.float32) * float(np.prod(sp))

    gap = GapRegion(
        id=1, volume_mm3=3200.0, volume_percent=12.5,
        centroid_voxels=(28.0, 32.0, 16.0),
        centroid_mm=(42.0, 32.0, 16.0),
        bounding_box=(25, 20, 10, 32, 44, 22),
        axial_extent_mm=10.5,
        max_cross_section_mm2=120.0,
        anatomical_zone="diaphysis",
        shape_descriptor="transverse",
    )
    mr = MissingBoneReport(
        intact_volume_mm3    = 25600.0,
        fractured_volume_mm3 = 22400.0,
        missing_volume_mm3   = 3200.0,
        missing_percent      = 12.5,
        missing_mask         = missing,
        centroid_mm          = (42.0, 32.0, 16.0),
        gap_regions          = [gap],
        n_gaps               = 1,
        axial_profile        = missing.sum(axis=(1, 2)).astype(np.float32),
        axial_profile_mm3    = axial,
        primary_fracture_zone = "diaphysis",
        total_axial_gap_mm   = 10.5,
        severity             = "moderate",
    )

    frags = [
        BoneFragment(1, 12000.0, (18.0, 32.0, 16.0), (0,10,5,25,54,27),  True,  "proximal"),
        BoneFragment(2, 10400.0, (54.0, 32.0, 16.0), (32,10,5,64,54,27), True,  "distal"),
    ]
    fr = FractureDetectionResult(
        n_fragments=2, fragments=frags,
        ao_code="A1", ao_description="Simple transverse fracture",
        fracture_pattern="transverse",
        fracture_z_voxel=28, fracture_z_mm=42.0,
        fracture_angle_deg=3.5,
        canal_diameter_mm=11.0,
        canal_measurements={"isthmus_diameter_mm": 11.0, "mean_canal_mm": 12.3, "femur_length_mm": 420.0},
        recommended_rod_diameter_mm=10.0,
        recommended_rod_length_mm=400.0,
        rod_sizing_confidence="high",
    )

    # Build demo meshes
    from inference.pipeline import mask_to_mesh
    meshes = {
        "fractured": mask_to_mesh(frac, sp),
        "completed": mask_to_mesh(comp, sp),
        "missing":   mask_to_mesh(missing, sp),
    }
    meshes = {k: v for k, v in meshes.items() if v is not None}

    from dataclasses import dataclass
    class R:
        fractured_mask  = frac
        completed_mask  = comp
        missing_mask    = missing
        missing_report  = mr
        fracture_result = fr
        spacing_mm      = sp
        meshes          = meshes
        timing          = {"demo": 0.0}
    return R()


# ═══════════════════════════════════════════════════════════════════════════
# Tab 1: Upload & Analyse
# ═══════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.header("📤 Upload CT Scan")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        upload_mode = st.radio(
            "Input type",
            ["DICOM folder (zip)", "NIfTI file (.nii / .nii.gz)", "Demo mode (no file needed)"],
            horizontal=True,
        )

        uploaded = None
        if upload_mode == "DICOM folder (zip)":
            uploaded = st.file_uploader(
                "Upload a ZIP file containing DICOM slices",
                type=["zip"],
            )
        elif upload_mode == "NIfTI file (.nii / .nii.gz)":
            uploaded = st.file_uploader(
                "Upload a NIfTI file",
                type=["nii", "gz"],
            )

    with col_r:
        st.info(
            "**Accepted formats**\n"
            "- DICOM series (zip of .dcm files)\n"
            "- NIfTI (.nii or .nii.gz)\n"
            "- MetaImage (.mha)\n\n"
            "**What the AI does**\n"
            "1. Segment the fractured femur\n"
            "2. Reconstruct the intact bone\n"
            "3. Measure missing bone volume\n"
            "4. Classify fracture (AO/OTA)\n"
            "5. Recommend IM rod size"
        )

    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        result = None
        with st.spinner("Running AI pipeline… this may take a minute"):
            if upload_mode == "Demo mode (no file needed)":
                result = _make_demo_result()
                st.success("✅ Demo analysis complete!")

            elif uploaded is not None and pipeline is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    if upload_mode == "DICOM folder (zip)":
                        zip_path = Path(tmpdir) / "scan.zip"
                        zip_path.write_bytes(uploaded.read())
                        with zipfile.ZipFile(zip_path) as zf:
                            zf.extractall(tmpdir)
                        scan_path = tmpdir
                    else:
                        suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
                        nii_path = Path(tmpdir) / f"scan{suffix}"
                        nii_path.write_bytes(uploaded.read())
                        scan_path = str(nii_path)

                    try:
                        result = pipeline.run(scan_path)
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")

            elif pipeline is None:
                st.warning("No model loaded — running demo mode instead")
                result = _make_demo_result()
                st.success("✅ Demo analysis complete!")

        if result is not None:
            st.session_state["result"] = result

    # ── Show quick metrics if result exists ───────────────────────────────
    if "result" in st.session_state:
        r  = st.session_state["result"]
        mr = r.missing_report
        fr = r.fracture_result

        st.divider()
        st.subheader("📊 Quick Summary")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Missing Volume",    f"{mr.missing_volume_mm3:.0f} mm³")
        c2.metric("Missing %",         f"{mr.missing_percent:.1f}%")
        c3.metric("Severity",          mr.severity.upper())
        c4.metric("Fracture Type",     fr.ao_code)
        c5.metric("Fragments",         fr.n_fragments)

        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Canal Diameter",    f"{fr.canal_diameter_mm:.1f} mm")
        c7.metric("Rod Diameter",      f"{fr.recommended_rod_diameter_mm} mm")
        c8.metric("Rod Length",        f"{fr.recommended_rod_length_mm} mm")
        c9.metric("Sizing Confidence", fr.rod_sizing_confidence.upper())


# ═══════════════════════════════════════════════════════════════════════════
# Tab 2: 3D Viewer
# ═══════════════════════════════════════════════════════════════════════════

with tab_3d:
    st.header("🧊 3D Reconstruction")

    if "result" not in st.session_state:
        st.info("Run an analysis in the Upload tab first.")
    else:
        r = st.session_state["result"]
        from visualization.renderer import FemurRenderer

        renderer = FemurRenderer(spacing_mm=r.spacing_mm)
        view_mode = st.radio(
            "View", ["Reconstruction + Missing Bone", "Fragment View"],
            horizontal=True,
        )

        if view_mode == "Reconstruction + Missing Bone":
            fig = renderer.render_reconstruction(
                r.meshes, r.missing_report, r.fracture_result,
                show_rod=show_rod, show_completed=show_ghost,
            )
        else:
            fig = renderer.render_fragments(r.fractured_mask, r.fracture_result)

        st.plotly_chart(fig, use_container_width=True)

        # Download STL buttons
        st.subheader("⬇️ Download Meshes")
        cols = st.columns(3)
        for i, (name, label) in enumerate([
            ("fractured", "Fractured bone"),
            ("completed", "Intact (reconstructed)"),
            ("missing",   "Missing region"),
        ]):
            if name in r.meshes:
                buf = io.BytesIO()
                r.meshes[name].export(buf, file_type="stl")
                cols[i].download_button(
                    label=f"💾 {label} (.stl)",
                    data=buf.getvalue(),
                    file_name=f"femur_{name}.stl",
                    mime="application/octet-stream",
                )


# ═══════════════════════════════════════════════════════════════════════════
# Tab 3: Missing Bone
# ═══════════════════════════════════════════════════════════════════════════

with tab_missing:
    st.header("🔴 Missing Bone Analysis")

    if "result" not in st.session_state:
        st.info("Run an analysis in the Upload tab first.")
    else:
        r  = st.session_state["result"]
        mr = r.missing_report
        from visualization.renderer import FemurRenderer

        # Volume breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Volume Summary")
            vol_data = {
                "Component": ["Intact (reconstructed)", "Fractured (present)", "Missing"],
                "Volume (mm³)": [
                    f"{mr.intact_volume_mm3:.1f}",
                    f"{mr.fractured_volume_mm3:.1f}",
                    f"{mr.missing_volume_mm3:.1f}",
                ],
                "Percentage": [
                    "100.0%",
                    f"{mr.fractured_volume_mm3 / mr.intact_volume_mm3 * 100:.1f}%",
                    f"{mr.missing_percent:.1f}%",
                ],
            }
            st.dataframe(vol_data, use_container_width=True, hide_index=True)

            st.subheader("Gap Regions")
            if mr.gap_regions:
                rows = []
                for g in mr.gap_regions:
                    rows.append({
                        "Gap #": g.id,
                        "Zone": g.anatomical_zone,
                        "Shape": g.shape_descriptor,
                        "Volume (mm³)": f"{g.volume_mm3:.0f}",
                        "Axial extent": f"{g.axial_extent_mm:.1f} mm",
                        "Max cross-sec": f"{g.max_cross_section_mm2:.0f} mm²",
                        "% of intact": f"{g.volume_percent:.1f}%",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.success("No significant bone gaps detected.")

        with col2:
            # Axial profile
            renderer = FemurRenderer(spacing_mm=r.spacing_mm)
            fig = renderer.render_axial_profile(mr)
            st.plotly_chart(fig, use_container_width=True)

            # Donut chart
            vals = [mr.fractured_volume_mm3, mr.missing_volume_mm3]
            fig2 = go.Figure(go.Pie(
                labels=["Present bone", "Missing bone"],
                values=vals,
                hole=0.55,
                marker=dict(colors=["rgba(100,140,200,0.8)", "rgba(255,60,40,0.85)"]),
            ))
            fig2.update_layout(
                title="Bone Volume Breakdown",
                paper_bgcolor="rgb(15,15,25)",
                font=dict(color="white"),
                showlegend=True,
                height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Tab 4: IM Rod Sizing
# ═══════════════════════════════════════════════════════════════════════════

with tab_rod:
    st.header("📏 IM Rod Sizing & Fracture Classification")

    if "result" not in st.session_state:
        st.info("Run an analysis in the Upload tab first.")
    else:
        r  = st.session_state["result"]
        fr = r.fracture_result

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("AO/OTA Classification")
            st.markdown(f"""
| Field | Value |
|---|---|
| **Code** | `{fr.ao_code}` |
| **Description** | {fr.ao_description} |
| **Pattern** | {fr.fracture_pattern} |
| **Fragments** | {fr.n_fragments} |
| **Angle** | {fr.fracture_angle_deg:.1f}° |
""")

            st.subheader("Fragment Table")
            frag_rows = [
                {
                    "Label": f.label,
                    "Volume (mm³)": f"{f.volume_mm3:.0f}",
                    "Centroid Z (mm)": f"{f.centroid_mm[0]:.1f}",
                    "Main fragment": "✅" if f.is_main else "—",
                }
                for f in fr.fragments
            ]
            st.dataframe(frag_rows, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("Medullary Canal Measurements")
            canal = fr.canal_measurements
            st.markdown(f"""
| Measurement | Value |
|---|---|
| **Isthmus diameter** | **{canal.get('isthmus_diameter_mm', '—')} mm** |
| Mean canal diameter | {canal.get('mean_canal_mm', '—')} mm |
| Femur length | {canal.get('femur_length_mm', '—')} mm |
""")

            conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
                fr.rod_sizing_confidence, "⚪"
            )
            st.subheader("IM Rod Recommendation")
            st.markdown(f"""
<div style="background: rgba(220,180,50,0.15); border: 2px solid rgba(220,180,50,0.6);
border-radius:10px; padding:20px; text-align:center">
<h2 style="color:#dca832; margin:0">
  ⌀ {fr.recommended_rod_diameter_mm} mm  ×  {fr.recommended_rod_length_mm} mm
</h2>
<p style="color:#ccc; margin:6px 0 0 0">
  Diameter × Length &nbsp;|&nbsp; Confidence: {conf_color} {fr.rod_sizing_confidence.upper()}
</p>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
**Sizing rationale**
- **Diameter** = isthmus canal diameter − 1 mm (0.5 mm clearance each side)
- **Length**   = femur length − 20 mm (distal lock clearance)

> ⚠️ Always confirm with fluoroscopy and templating before surgery.
""")


# ═══════════════════════════════════════════════════════════════════════════
# Tab 5: Report
# ═══════════════════════════════════════════════════════════════════════════

with tab_report:
    st.header("📄 Full Analysis Report")

    if "result" not in st.session_state:
        st.info("Run an analysis in the Upload tab first.")
    else:
        r = st.session_state["result"]

        from analysis.missing_bone_analyzer import MissingBoneAnalyzer
        from analysis.fracture_detector import FractureDetector

        missing_txt = MissingBoneAnalyzer(r.spacing_mm).format_report(r.missing_report)
        fracture_txt = FractureDetector(r.spacing_mm).format_report(r.fracture_result)
        summary_txt  = r.summary() if hasattr(r, 'summary') else ""

        full_report = "\n\n".join([summary_txt, missing_txt, fracture_txt])

        st.code(full_report, language=None)

        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=full_report,
            file_name="femur_analysis_report.txt",
            mime="text/plain",
            type="primary",
        )