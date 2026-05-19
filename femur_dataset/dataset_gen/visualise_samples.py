"""
dataset_gen/visualise_samples.py
Visualise generated dataset samples.

Produces a grid figure showing for each sample:
  Row 1 — axial (z) CT slices    (windowed HU)
  Row 2 — axial intact mask slices
  Row 3 — axial fractured mask slices
  Row 4 — axial gap (missing bone) slices
  Row 5 — coronal MIP of intact vs fractured

Usage
-----
  python -m dataset_gen.visualise_samples \
      --data_dir data/processed \
      --n_samples 6 \
      --output preview.png
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless / no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# ── custom colourmaps ────────────────────────────────────────────────
_BONE_CMAP    = plt.cm.bone
_MISSING_CMAP = LinearSegmentedColormap.from_list(
    'missing', [(1,1,1,0), (0.87, 0.18, 0.12, 0.85)], N=256
)
_MASK_CMAP    = LinearSegmentedColormap.from_list(
    'mask', [(1,1,1,0), (0.18, 0.37, 0.63, 0.80)], N=256
)


def _mid_slice(vol, axis=0):
    """Return the middle slice along *axis*."""
    idx = vol.shape[axis] // 2
    return np.take(vol, idx, axis=axis)


def _mip(vol, axis=0):
    """Maximum intensity projection along *axis*."""
    return vol.max(axis=axis)


def visualise_sample(npz_path: str) -> plt.Figure:
    """
    Produce a detailed single-sample figure.

    Rows: CT axial | intact mask axial | fractured mask axial |
          gap mask axial | coronal MIP comparison
    """
    data = np.load(npz_path, allow_pickle=True)

    windowed  = data['windowed']         # (D,H,W)
    intact    = data['mask']
    fractured = data['fractured']
    gap       = data['gap_mask']

    ao_code       = str(data.get('ao_code',       '?'))
    fracture_type = str(data.get('fracture_type', '?'))
    gap_mm        = float(data.get('gap_size_mm',  0))
    n_frags       = int(data.get('n_fragments',    0))

    # pick representative axial slices (25%, 50%, 75% of depth)
    D = windowed.shape[0]
    z_slices = [int(D * 0.25), int(D * 0.50), int(D * 0.75)]

    fig = plt.figure(figsize=(14, 10), facecolor='#0D1117')
    fig.suptitle(
        f"Sample: {Path(npz_path).stem}   |   AO: {ao_code}   "
        f"Type: {fracture_type}   Gap: {gap_mm:.1f} mm   Fragments: {n_frags}",
        color='white', fontsize=11, y=0.98,
    )

    n_cols  = len(z_slices) + 1   # slices + MIP
    n_rows  = 4
    gs      = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                hspace=0.08, wspace=0.04,
                                left=0.04, right=0.96, top=0.94, bottom=0.04)

    row_labels = ['CT (windowed)', 'Intact mask', 'Fractured mask', 'Missing bone']
    layer_data = [windowed, intact, fractured, gap]
    cmaps      = [_BONE_CMAP, _MASK_CMAP, _MASK_CMAP, _MISSING_CMAP]

    for row_i, (layer, cmap, rlabel) in enumerate(zip(layer_data, cmaps, row_labels)):
        for col_i, z in enumerate(z_slices):
            ax = fig.add_subplot(gs[row_i, col_i])
            sl = layer[z]
            ax.imshow(sl, cmap=cmap, interpolation='nearest', aspect='equal')
            ax.axis('off')
            if col_i == 0:
                ax.set_ylabel(rlabel, color='white', fontsize=8)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])
            if row_i == 0:
                ax.set_title(f'z={z}', color='#88A4C8', fontsize=8)

        # Coronal MIP column
        ax_mip = fig.add_subplot(gs[row_i, n_cols - 1])
        mip    = _mip(layer, axis=2)    # collapse along W → (D,H)
        ax_mip.imshow(mip, cmap=cmap, interpolation='bilinear', aspect='auto')
        ax_mip.axis('off')
        if row_i == 0:
            ax_mip.set_title('Coronal MIP', color='#88A4C8', fontsize=8)

    # Overlay gap on fractured CT in MIP
    ax_overlay = fig.add_subplot(gs[2, n_cols - 1])
    ax_overlay.imshow(_mip(fractured, axis=2), cmap=_MASK_CMAP, aspect='auto')
    ax_overlay.imshow(_mip(gap, axis=2),       cmap=_MISSING_CMAP, alpha=0.65, aspect='auto')
    ax_overlay.axis('off')
    ax_overlay.set_title('Frac + Gap', color='#E55347', fontsize=8)

    return fig


def visualise_grid(
    data_dir:  str,
    n_samples: int  = 6,
    seed:      int  = 0,
    output:    str  = 'preview.png',
):
    """
    Produce a summary grid showing N samples side by side.
    Each column = one sample; rows = CT / intact / fractured / gap.
    """
    files = sorted(Path(data_dir).glob('sample_*.npz'))
    if not files:
        print(f"No .npz files found in {data_dir}")
        return

    random.seed(seed)
    chosen = random.sample(files, min(n_samples, len(files)))
    chosen.sort()

    n_cols = len(chosen)
    n_rows = 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.8),
        facecolor='#0D1117',
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    row_labels = ['CT (axial mid)', 'Intact', 'Fractured', 'Missing bone']
    cmaps      = [_BONE_CMAP, _MASK_CMAP, _MASK_CMAP, _MISSING_CMAP]

    for col_i, fpath in enumerate(chosen):
        data      = np.load(fpath, allow_pickle=True)
        windowed  = data['windowed']
        intact    = data['mask']
        fractured = data['fractured']
        gap       = data['gap_mask']
        layers    = [windowed, intact, fractured, gap]

        ao_code = str(data.get('ao_code', '?'))
        gap_mm  = float(data.get('gap_size_mm', 0))

        for row_i, (layer, cmap) in enumerate(zip(layers, cmaps)):
            ax = axes[row_i][col_i]
            sl = _mid_slice(layer, axis=0)
            ax.imshow(sl, cmap=cmap, interpolation='nearest', aspect='equal')
            ax.axis('off')
            if row_i == 0:
                ax.set_title(
                    f"{fpath.stem}\nAO:{ao_code}  {gap_mm:.1f}mm",
                    color='white', fontsize=7.5, pad=3,
                )
            if col_i == 0:
                ax.set_ylabel(row_labels[row_i], color='#88A4C8', fontsize=8)
                ax.yaxis.set_visible(True)
                ax.set_yticks([])

    plt.suptitle(
        f'Synthetic Femur Fracture Dataset — {n_cols} samples',
        color='white', fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close(fig)
    print(f"Saved preview → {output}")


def print_dataset_stats(data_dir: str):
    """Print a quick statistical summary of the dataset."""
    from collections import Counter
    files = sorted(Path(data_dir).glob('sample_*.npz'))
    if not files:
        print("No samples found.")
        return

    ao_codes  = Counter()
    gap_sizes = []
    n_frags   = []

    for f in files:
        d = np.load(f, allow_pickle=True)
        ao_codes[str(d.get('ao_code', '?'))] += 1
        gap_sizes.append(float(d.get('gap_size_mm', 0)))
        n_frags.append(int(d.get('n_fragments', 0)))

    print(f"\n{'─'*50}")
    print(f"  Dataset Stats — {len(files)} samples")
    print(f"{'─'*50}")
    print(f"  Mean gap size    : {np.mean(gap_sizes):.2f} mm")
    print(f"  Std  gap size    : {np.std(gap_sizes):.2f} mm")
    print(f"  Min  gap size    : {np.min(gap_sizes):.2f} mm")
    print(f"  Max  gap size    : {np.max(gap_sizes):.2f} mm")
    print(f"  Mean fragments   : {np.mean(n_frags):.2f}")
    print(f"\n  AO Code distribution:")
    for code, cnt in sorted(ao_codes.items()):
        bar = '█' * (cnt * 25 // max(ao_codes.values()))
        print(f"    {code:4s}  {cnt:5d}  {bar}")
    print(f"{'─'*50}\n")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  type=str, default='data/processed')
    parser.add_argument('--n_samples', type=int, default=6)
    parser.add_argument('--seed',      type=int, default=0)
    parser.add_argument('--output',    type=str, default='dataset_preview.png')
    parser.add_argument('--stats',     action='store_true', help='Print stats only')
    args = parser.parse_args()

    if args.stats:
        print_dataset_stats(args.data_dir)
    else:
        print_dataset_stats(args.data_dir)
        visualise_grid(args.data_dir, args.n_samples, args.seed, args.output)