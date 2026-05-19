"""
visualization/renderer.py — 3-D femur visualisation with Plotly.

Renders three overlapping meshes in a single interactive 3-D figure:
    1. Fractured bone  — blue/grey, semi-transparent
    2. Completed bone  — light ghost overlay, very transparent
    3. Missing region  — vivid red/orange, opaque

Also produces:
    - Axial profile bar chart (missing bone per slice)
    - Fragment view (each fragment a different colour)
    - IM rod overlay (cylinder positioned in the canal)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import trimesh
from plotly.subplots import make_subplots


# ═══════════════════════════════════════════════════════════════════════════
# Colour palette
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "fractured":  "rgba(100, 140, 200, 0.55)",   # steely blue
    "completed":  "rgba(200, 220, 255, 0.15)",   # ghost white
    "missing":    "rgba(255,  60,  40, 0.90)",   # vivid red
    "fragment_0": "rgba( 70, 130, 180, 0.80)",   # proximal — steel blue
    "fragment_1": "rgba( 46, 160,  80, 0.80)",   # distal   — green
    "rod":        "rgba(220, 180,  50, 0.85)",   # rod      — gold
}

FRAGMENT_PALETTE = [
    "rgba(70,130,180,0.8)",
    "rgba(46,160,80,0.8)",
    "rgba(200,80,50,0.8)",
    "rgba(160,90,200,0.8)",
    "rgba(200,170,40,0.8)",
    "rgba(60,190,200,0.8)",
]


# ═══════════════════════════════════════════════════════════════════════════
# Mesh → Plotly Mesh3d
# ═══════════════════════════════════════════════════════════════════════════

def mesh_to_plotly(
    mesh:  trimesh.Trimesh,
    name:  str,
    color: str,
    show_edges: bool = False,
) -> go.Mesh3d:
    v = mesh.vertices
    f = mesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        name=name,
        color=color,
        flatshading=False,
        lighting=dict(
            ambient=0.5, diffuse=0.8,
            specular=0.3, roughness=0.5,
        ),
        lightposition=dict(x=100, y=200, z=150),
        showscale=False,
        hoverinfo="name",
    )


# ═══════════════════════════════════════════════════════════════════════════
# IM Rod cylinder
# ═══════════════════════════════════════════════════════════════════════════

def make_rod_mesh(
    center_mm:   Tuple[float, float, float],
    diameter_mm: float,
    length_mm:   float,
    n_sides:     int = 20,
) -> go.Mesh3d:
    """Create a Plotly Mesh3d cylinder representing the IM rod."""
    r   = diameter_mm / 2.0
    cz, cy, cx = center_mm
    z0 = cz - length_mm / 2.0
    z1 = cz + length_mm / 2.0

    theta  = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    xs     = cx + r * np.cos(theta)
    ys     = cy + r * np.sin(theta)

    # Vertices: bottom ring + top ring + two cap centres
    verts_bottom = np.column_stack([np.full(n_sides, z0), ys, xs])
    verts_top    = np.column_stack([np.full(n_sides, z1), ys, xs])
    cap_bot = np.array([[z0, cy, cx]])
    cap_top = np.array([[z1, cy, cx]])

    verts = np.vstack([verts_bottom, verts_top, cap_bot, cap_top])
    bot_c_idx = 2 * n_sides
    top_c_idx = 2 * n_sides + 1

    faces = []
    for i in range(n_sides):
        j = (i + 1) % n_sides
        # Side quad → 2 triangles
        faces.append([i, j, i + n_sides])
        faces.append([j, j + n_sides, i + n_sides])
        # Bottom cap
        faces.append([bot_c_idx, j, i])
        # Top cap
        faces.append([top_c_idx, i + n_sides, j + n_sides])

    faces = np.array(faces)
    v = verts
    f = faces
    return go.Mesh3d(
        x=v[:, 2], y=v[:, 1], z=v[:, 0],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        name=f"IM Rod ({diameter_mm}mm × {length_mm}mm)",
        color=COLORS["rod"],
        flatshading=True,
        opacity=0.85,
        hoverinfo="name",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main renderer
# ═══════════════════════════════════════════════════════════════════════════

class FemurRenderer:
    """
    Parameters
    ----------
    spacing_mm : voxel spacing — used to place annotation text correctly
    """

    def __init__(self, spacing_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.spacing = spacing_mm

    # ── Full reconstruction view ───────────────────────────────────────────

    def render_reconstruction(
        self,
        meshes:           Dict[str, trimesh.Trimesh],
        missing_report,                              # MissingBoneReport
        fracture_result,                             # FractureDetectionResult
        show_rod:         bool = True,
        show_completed:   bool = True,
    ) -> go.Figure:
        """
        Main 3-D reconstruction figure with:
          - Fractured bone
          - Ghost intact bone
          - Red missing region
          - Gold IM rod cylinder
        """
        traces = []

        if "fractured" in meshes:
            traces.append(mesh_to_plotly(meshes["fractured"], "Fractured bone", COLORS["fractured"]))

        if show_completed and "completed" in meshes:
            traces.append(mesh_to_plotly(meshes["completed"], "Intact (reconstructed)", COLORS["completed"]))

        if "missing" in meshes:
            traces.append(mesh_to_plotly(meshes["missing"], "Missing bone", COLORS["missing"]))

        # IM Rod
        if show_rod and fracture_result.recommended_rod_diameter_mm:
            centroid = missing_report.centroid_mm or (0.0, 0.0, 0.0)
            rod = make_rod_mesh(
                center_mm   = centroid,
                diameter_mm = fracture_result.recommended_rod_diameter_mm,
                length_mm   = fracture_result.recommended_rod_length_mm or 400.0,
            )
            traces.append(rod)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=(
                    f"Femur Reconstruction — {fracture_result.ao_code} "
                    f"| Missing: {missing_report.missing_percent:.1f}% "
                    f"| Severity: {missing_report.severity.upper()}"
                ),
                x=0.5, font_size=14,
            ),
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="Z (mm)",
                bgcolor="rgb(15,15,25)",
                xaxis=dict(backgroundcolor="rgb(20,20,30)", gridcolor="rgb(60,60,80)"),
                yaxis=dict(backgroundcolor="rgb(20,20,30)", gridcolor="rgb(60,60,80)"),
                zaxis=dict(backgroundcolor="rgb(20,20,30)", gridcolor="rgb(60,60,80)"),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
                aspectmode="data",
            ),
            paper_bgcolor="rgb(15,15,25)",
            plot_bgcolor ="rgb(15,15,25)",
            font=dict(color="white"),
            legend=dict(
                bgcolor="rgba(30,30,50,0.8)",
                bordercolor="rgba(100,100,150,0.5)",
                borderwidth=1,
            ),
            margin=dict(l=0, r=0, t=50, b=0),
            height=650,
        )
        return fig

    # ── Fragment view ─────────────────────────────────────────────────────

    def render_fragments(
        self,
        fractured_mask: np.ndarray,
        fracture_result,
    ) -> go.Figure:
        """Colour each bone fragment differently."""
        from skimage.measure import label as cc_label, marching_cubes

        labeled = cc_label((fractured_mask > 0.5).astype(np.uint8))
        traces  = []

        for i, frag in enumerate(fracture_result.fragments):
            frag_mask = (labeled == (i + 1)).astype(np.float32)
            if frag_mask.sum() < 100:
                continue
            padded = np.pad(frag_mask, 1)
            try:
                v, f, _, _ = marching_cubes(padded, level=0.5, spacing=self.spacing)
                v -= np.array(self.spacing)
                color = FRAGMENT_PALETTE[i % len(FRAGMENT_PALETTE)]
                traces.append(go.Mesh3d(
                    x=v[:, 2], y=v[:, 1], z=v[:, 0],
                    i=f[:, 0], j=f[:, 1], k=f[:, 2],
                    name=frag.label,
                    color=color,
                    flatshading=False,
                    opacity=0.85,
                    hoverinfo="name",
                    lighting=dict(ambient=0.5, diffuse=0.8),
                ))
            except Exception:
                continue

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=f"Fragment View — {fracture_result.n_fragments} fragments ({fracture_result.ao_code})",
                x=0.5,
            ),
            scene=dict(
                aspectmode="data",
                bgcolor="rgb(15,15,25)",
                xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
            ),
            paper_bgcolor="rgb(15,15,25)",
            font=dict(color="white"),
            height=600,
        )
        return fig

    # ── Axial profile chart ────────────────────────────────────────────────

    def render_axial_profile(self, missing_report) -> go.Figure:
        """Bar chart showing missing bone volume per axial slice."""
        prof = missing_report.axial_profile_mm3
        if prof is None:
            return go.Figure()

        z_mm = np.arange(len(prof)) * float(self.spacing[0])

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=z_mm,
            y=prof,
            name="Missing bone (mm³/slice)",
            marker_color=[
                f"rgba(255,{max(60, int(60 + 180 * (1 - v / (prof.max() + 1e-6))))},40,0.85)"
                for v in prof
            ],
        ))

        # Mark fracture zone
        if missing_report.gap_regions:
            for gap in missing_report.gap_regions:
                z0_vox, *_, z1_vox, *_ = gap.bounding_box
                fig.add_vrect(
                    x0=z0_vox * self.spacing[0],
                    x1=z1_vox * self.spacing[0],
                    fillcolor="rgba(255,100,100,0.15)",
                    line_width=0,
                    annotation_text=f"Gap #{gap.id}",
                    annotation_font_color="white",
                )

        fig.update_layout(
            title="Missing Bone — Axial Profile",
            xaxis_title="Axial position (mm)",
            yaxis_title="Missing volume (mm³ / slice)",
            paper_bgcolor="rgb(15,15,25)",
            plot_bgcolor ="rgb(20,20,35)",
            font=dict(color="white"),
            xaxis=dict(gridcolor="rgb(60,60,80)"),
            yaxis=dict(gridcolor="rgb(60,60,80)"),
            height=350,
        )
        return fig

    # ── Dashboard (all panels) ────────────────────────────────────────────

    def build_dashboard(
        self,
        meshes:          Dict[str, trimesh.Trimesh],
        missing_report,
        fracture_result,
    ) -> Dict[str, go.Figure]:
        """Return all figures in a dict — let the UI decide how to lay them out."""
        return {
            "reconstruction": self.render_reconstruction(meshes, missing_report, fracture_result),
            "fragments":      self.render_fragments(
                                  # fallback empty array if no mask available
                                  np.zeros((1, 1, 1)), fracture_result
                              ),
            "axial_profile":  self.render_axial_profile(missing_report),
        }