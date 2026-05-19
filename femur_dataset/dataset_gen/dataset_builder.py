"""
dataset_gen/dataset_builder.py
Batch synthetic femur dataset generator.

For each sample the pipeline is:
  1. Generate intact 3-D femur geometry   (FemurGeometryGenerator)
  2. Apply a synthetic fracture           (FractureGenerator)
  3. Simulate CT HU values + noise        (CTSimulator)
  4. Preprocess (window, resample, crop)  (preprocessor.preprocess)
  5. Save .npz file                       (numpy compressed)

Each .npz file contains:
  windowed      : (D,H,W) float32  — normalised CT volume [0,1]
  mask          : (D,H,W) float32  — binary intact femur mask (seg target)
  fractured     : (D,H,W) float32  — binary fractured mask   (completion input)
  completed     : (D,H,W) float32  — binary intact mask       (completion target)
  gap_mask      : (D,H,W) float32  — missing bone voxels
  spacing       : (3,)    float32  — voxel spacing mm [z,y,x]
  ao_code       : str
  fracture_type : str
  gap_size_mm   : float
  n_fragments   : int
  femur_params  : dict   (head radius, canal diameter, femur length, …)

Usage
-----
  python -m dataset_gen.dataset_builder \
      --n_samples 500 \
      --output_dir data/processed \
      --volume_shape 128 128 64 \
      --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

# Make sure the repo root is on the path when running standalone
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_gen.femur_geometry  import FemurGeometryGenerator
from dataset_gen.fracture_generator import FractureGenerator
from dataset_gen.ct_simulator    import CTSimulator


# ══════════════════════════════════════════════════════════════════════
# Per-sample generation (runs in subprocess for multiprocessing)
# ══════════════════════════════════════════════════════════════════════

def _generate_one(args_tuple) -> dict:
    """
    Generate one dataset sample.
    Returns a dict with status and metadata (mask arrays NOT returned —
    they are saved to disk directly to avoid pickling huge arrays).
    """
    (
        sample_idx,
        output_dir,
        volume_shape,
        target_spacing,
        hu_min, hu_max,
        fracture_type,  # None → random
    ) = args_tuple

    seed = sample_idx * 7919   # deterministic but spread out
    rng  = np.random.default_rng(seed)

    try:
        D, H, W = volume_shape

        # ── 1. Intact femur geometry ──────────────────────────────────
        left = bool(rng.integers(0, 2))
        geo  = FemurGeometryGenerator(volume_shape=(D*2, H, W), seed=seed, left_side=left)
        geo_result = geo.generate()

        intact    = geo_result['mask']        # (D*2, H, W)
        canal     = geo_result['canal_mask']
        params    = geo_result['params']
        dia_range = geo_result['z_diaphysis_range']

        # ── 2. Apply fracture ─────────────────────────────────────────
        frac_gen = FractureGenerator(voxel_spacing_mm=1.0, seed=seed + 1)
        frac_res = frac_gen.apply(
            intact_mask   = intact,
            fracture_type = fracture_type,
            diaphysis_range = dia_range,
        )

        # ── 3. Simulate CT ────────────────────────────────────────────
        sim = CTSimulator(voxel_spacing_mm=1.0, seed=seed + 2)
        # Simulate CT from fractured bone (what a real scan would show)
        ct_volume = sim.simulate(
            femur_mask = frac_res.fractured_mask,
            canal_mask = canal & frac_res.fractured_mask,
            add_noise         = True,
            add_bias_field    = bool(rng.integers(0, 2)),
            add_ring_artefact = rng.random() < 0.15,
            add_motion_blur   = rng.random() < 0.10,
        )

        # ── 4. Resample + window + crop ───────────────────────────────
        from data.preprocessor import preprocess
        target_shape_3d = (D, H, W)

        # Raw spacing assumed 1 mm isotropic (our generator space)
        raw_spacing = (1.0, 1.0, 1.0)
        prep = preprocess(
            ct_volume,
            raw_spacing,
            target_spacing = target_spacing,
            target_shape   = target_shape_3d,
            hu_min = hu_min,
            hu_max = hu_max,
        )

        # Crop / pad masks to match preprocessed volume shape
        from data.preprocessor import crop_or_pad
        intact_c,   _ = crop_or_pad(intact.astype(np.float32),              target_shape_3d)
        fractured_c, _= crop_or_pad(frac_res.fractured_mask.astype(np.float32), target_shape_3d)
        gap_c, _      = crop_or_pad(frac_res.gap_mask.astype(np.float32),    target_shape_3d)

        # ── 5. Save .npz ──────────────────────────────────────────────
        out_path = Path(output_dir) / f"sample_{sample_idx:05d}.npz"
        np.savez_compressed(
            str(out_path),
            # CT volumes
            windowed      = prep['windowed'],             # (D,H,W) float32 [0,1]
            # Segmentation targets
            mask          = intact_c.astype(np.float32),  # intact femur
            # Completion network inputs/targets
            fractured     = fractured_c.astype(np.float32),
            completed     = intact_c.astype(np.float32),
            gap_mask      = gap_c.astype(np.float32),
            # Metadata stored as 0-d arrays for easy loading
            spacing       = np.array(target_spacing, dtype=np.float32),
            gap_size_mm   = np.float32(frac_res.gap_size_mm),
            n_fragments   = np.int32(frac_res.n_fragments),
            tilt_angle    = np.float32(frac_res.tilt_angle_deg),
            left_side     = np.bool_(left),
        )

        return {
            'idx':          sample_idx,
            'status':       'ok',
            'ao_code':      frac_res.ao_code,
            'fracture_type':frac_res.fracture_type,
            'gap_size_mm':  round(frac_res.gap_size_mm, 2),
            'n_fragments':  frac_res.n_fragments,
            'femur_len_mm': params['femur_length_mm'],
            'canal_mm':     params['canal_diameter_mm'],
            'path':         str(out_path),
        }

    except Exception as e:
        import traceback
        return {
            'idx':    sample_idx,
            'status': 'error',
            'error':  str(e),
            'trace':  traceback.format_exc(),
        }


# ══════════════════════════════════════════════════════════════════════
# Builder class
# ══════════════════════════════════════════════════════════════════════

class DatasetBuilder:
    """
    Parameters
    ----------
    output_dir     : where to save .npz files
    volume_shape   : (D, H, W) final voxel grid for each sample
    target_spacing : (sz, sy, sx) mm — resample target
    hu_min/hu_max  : HU window for normalisation
    n_workers      : parallel processes (1 = serial, safe for debugging)
    """

    def __init__(
        self,
        output_dir:      str  = 'data/processed',
        volume_shape:    Tuple = (128, 128, 64),
        target_spacing:  Tuple = (1.5, 1.0, 1.0),
        hu_min:          float = 200.0,
        hu_max:          float = 1800.0,
        n_workers:       int   = 1,
    ):
        self.output_dir     = Path(output_dir)
        self.volume_shape   = tuple(volume_shape)
        self.target_spacing = tuple(target_spacing)
        self.hu_min         = hu_min
        self.hu_max         = hu_max
        self.n_workers      = n_workers
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── public ──────────────────────────────────────────────────────

    def build(
        self,
        n_samples:       int,
        fracture_dist:   dict | None = None,
        start_idx:       int = 0,
        resume:          bool = True,
    ) -> dict:
        """
        Generate `n_samples` synthetic CT + fracture samples.

        Parameters
        ----------
        n_samples      : total number of samples to generate
        fracture_dist  : dict mapping AO code → count, e.g.
                         {'A1':150,'A2':120,'A3':60,'B1':80,'B2':50,'C1':30,'C2':10}
                         If None, type is random per sample.
        start_idx      : starting sample index (useful for extending dataset)
        resume         : skip samples whose .npz already exists

        Returns
        -------
        summary dict with counts per fracture type and error list
        """
        print(f"\n{'═'*60}")
        print(f"  FemurRecon Synthetic Dataset Builder")
        print(f"{'═'*60}")
        print(f"  Samples       : {n_samples}")
        print(f"  Output dir    : {self.output_dir}")
        print(f"  Volume shape  : {self.volume_shape}")
        print(f"  Spacing (mm)  : {self.target_spacing}")
        print(f"  Workers       : {self.n_workers}")
        print(f"{'═'*60}\n")

        # Build list of (sample_idx, fracture_type) assignments
        assignments = self._build_assignments(n_samples, fracture_dist, start_idx)

        # Filter already-generated samples
        if resume:
            existing = {p.stem for p in self.output_dir.glob('sample_*.npz')}
            assignments = [
                a for a in assignments
                if f"sample_{a[0]:05d}" not in existing
            ]
            skipped = n_samples - len(assignments)
            if skipped > 0:
                print(f"  Resuming: skipping {skipped} already-generated samples\n")

        if not assignments:
            print("  All samples already generated!")
            return self._load_manifest()

        # Build argument tuples for worker function
        arg_tuples = [
            (
                idx,
                str(self.output_dir),
                self.volume_shape,
                self.target_spacing,
                self.hu_min,
                self.hu_max,
                ft,
            )
            for idx, ft in assignments
        ]

        # ── Run ──────────────────────────────────────────────────────
        t0       = time.time()
        metadata = []
        errors   = []

        if self.n_workers == 1:
            # Serial (easier to debug)
            for args in tqdm(arg_tuples, desc='Generating', unit='sample'):
                result = _generate_one(args)
                if result['status'] == 'ok':
                    metadata.append(result)
                else:
                    errors.append(result)
                    print(f"\n  ✗ Sample {result['idx']}: {result['error']}")
        else:
            with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
                futures = {pool.submit(_generate_one, a): a[0] for a in arg_tuples}
                with tqdm(total=len(futures), desc='Generating', unit='sample') as pbar:
                    for fut in as_completed(futures):
                        result = fut.result()
                        if result['status'] == 'ok':
                            metadata.append(result)
                        else:
                            errors.append(result)
                        pbar.update(1)

        elapsed = time.time() - t0
        # Sort by index
        metadata.sort(key=lambda x: x['idx'])

        # ── Save manifest ─────────────────────────────────────────────
        manifest_path = self.output_dir / 'manifest.json'
        # Merge with existing manifest if resuming
        existing_meta = []
        if manifest_path.exists():
            with open(manifest_path) as f:
                existing_meta = json.load(f).get('samples', [])
        all_meta = {m['idx']: m for m in existing_meta}
        all_meta.update({m['idx']: m for m in metadata})
        all_meta_list = sorted(all_meta.values(), key=lambda x: x['idx'])

        summary = self._compute_summary(all_meta_list, errors, elapsed)
        with open(manifest_path, 'w') as f:
            json.dump({'summary': summary, 'samples': all_meta_list}, f, indent=2)

        self._print_summary(summary)
        return summary

    # ── private ──────────────────────────────────────────────────────

    def _build_assignments(self, n, fracture_dist, start_idx):
        if fracture_dist is None:
            return [(start_idx + i, None) for i in range(n)]

        assignments = []
        idx = start_idx
        for ao_code, count in fracture_dist.items():
            for _ in range(count):
                assignments.append((idx, ao_code))
                idx += 1
        # Fill remainder randomly if counts < n
        while len(assignments) < n:
            assignments.append((idx, None))
            idx += 1
        return assignments[:n]

    def _compute_summary(self, metadata, errors, elapsed):
        from collections import Counter
        types  = Counter(m['fracture_type'] for m in metadata)
        ao     = Counter(m['ao_code']       for m in metadata)
        return {
            'total_samples':    len(metadata),
            'total_errors':     len(errors),
            'elapsed_s':        round(elapsed, 1),
            'per_type':         dict(types),
            'per_ao_code':      dict(ao),
            'mean_gap_mm':      round(float(np.mean([m['gap_size_mm'] for m in metadata])), 2) if metadata else 0,
            'mean_femur_len_mm':round(float(np.mean([m['femur_len_mm'] for m in metadata])), 1) if metadata else 0,
        }

    def _load_manifest(self):
        path = self.output_dir / 'manifest.json'
        if path.exists():
            with open(path) as f:
                return json.load(f).get('summary', {})
        return {}

    def _print_summary(self, s):
        print(f"\n{'═'*60}")
        print(f"  Dataset Generation Complete")
        print(f"{'═'*60}")
        print(f"  Samples generated : {s['total_samples']}")
        print(f"  Errors            : {s['total_errors']}")
        print(f"  Time              : {s['elapsed_s']} s")
        print(f"  Mean gap size     : {s['mean_gap_mm']} mm")
        print(f"  Mean femur length : {s['mean_femur_len_mm']} mm")
        print(f"\n  Fracture type distribution:")
        for ft, cnt in sorted(s['per_type'].items(), key=lambda x: -x[1]):
            bar = '█' * (cnt * 30 // max(s['per_type'].values()))
            print(f"    {ft:25s} {cnt:4d}  {bar}")
        print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic femur fracture dataset')
    parser.add_argument('--n_samples',    type=int,   default=500)
    parser.add_argument('--output_dir',   type=str,   default='data/processed')
    parser.add_argument('--volume_shape', type=int,   nargs=3, default=[128, 128, 64],
                        metavar=('D','H','W'))
    parser.add_argument('--spacing',      type=float, nargs=3, default=[1.5, 1.0, 1.0],
                        metavar=('Z','Y','X'))
    parser.add_argument('--workers',      type=int,   default=1)
    parser.add_argument('--start_idx',    type=int,   default=0)
    parser.add_argument('--no_resume',    action='store_true')

    # Optional per-type distribution
    parser.add_argument('--dist_A1', type=int, default=None)
    parser.add_argument('--dist_A2', type=int, default=None)
    parser.add_argument('--dist_A3', type=int, default=None)
    parser.add_argument('--dist_B1', type=int, default=None)
    parser.add_argument('--dist_B2', type=int, default=None)
    parser.add_argument('--dist_C1', type=int, default=None)
    parser.add_argument('--dist_C2', type=int, default=None)

    args = parser.parse_args()

    # Build fracture distribution dict if any counts specified
    dist_args = {k: getattr(args, f'dist_{k}') for k in ['A1','A2','A3','B1','B2','C1','C2']}
    fracture_dist = {k: v for k, v in dist_args.items() if v is not None} or None

    builder = DatasetBuilder(
        output_dir     = args.output_dir,
        volume_shape   = tuple(args.volume_shape),
        target_spacing = tuple(args.spacing),
        n_workers      = args.workers,
    )
    builder.build(
        n_samples      = args.n_samples,
        fracture_dist  = fracture_dist,
        start_idx      = args.start_idx,
        resume         = not args.no_resume,
    )