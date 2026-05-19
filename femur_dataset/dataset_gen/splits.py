"""
dataset_gen/splits.py
Create train / val / test split JSON from a generated dataset manifest.

Stratified by fracture type so every split has a balanced distribution.

Usage
-----
  python -m dataset_gen.splits \
      --manifest data/processed/manifest.json \
      --output   data/processed/splits.json \
      --train 0.75 --val 0.15 --test 0.10 \
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def make_splits(
    manifest_path: str,
    output_path:   str,
    ratios:        Tuple[float, float, float] = (0.75, 0.15, 0.10),
    seed:          int = 42,
) -> Dict[str, List[str]]:
    """
    Parameters
    ----------
    manifest_path : path to manifest.json produced by DatasetBuilder
    output_path   : where to write splits.json
    ratios        : (train, val, test) fractions — must sum to 1
    seed          : for reproducibility

    Returns
    -------
    dict with keys 'train', 'val', 'test' → lists of sample stem names
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1"

    with open(manifest_path) as f:
        data = json.load(f)
    samples = data.get('samples', [])

    if not samples:
        raise ValueError(f"No samples found in {manifest_path}")

    # Group by fracture type for stratification
    by_type: Dict[str, List[str]] = defaultdict(list)
    for s in samples:
        if s.get('status') == 'ok':
            stem = f"sample_{s['idx']:05d}"
            by_type[s['fracture_type']].append(stem)

    rng = random.Random(seed)
    splits: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}

    for ftype, names in by_type.items():
        rng.shuffle(names)
        n      = len(names)
        n_val  = max(1, round(ratios[1] * n))
        n_test = max(1, round(ratios[2] * n))
        n_train = n - n_val - n_test

        splits['train'].extend(names[:n_train])
        splits['val'].extend(names[n_train: n_train + n_val])
        splits['test'].extend(names[n_train + n_val:])

    # Shuffle within each split
    for key in splits:
        rng.shuffle(splits[key])

    # Print summary
    total = sum(len(v) for v in splits.values())
    print(f"\n{'─'*50}")
    print(f"  Dataset Split Summary")
    print(f"{'─'*50}")
    for key, names in splits.items():
        print(f"  {key:6s}: {len(names):5d}  ({len(names)/total*100:.1f}%)")
    print(f"  {'Total':6s}: {total:5d}")
    print(f"{'─'*50}")
    print(f"\n  Fracture type distribution per split:")

    by_type_split: Dict[str, Dict[str, int]] = {k: defaultdict(int) for k in splits}
    name_to_type  = {f"sample_{s['idx']:05d}": s['fracture_type'] for s in samples if s.get('status')=='ok'}
    for split_name, names in splits.items():
        for n in names:
            by_type_split[split_name][name_to_type.get(n, 'unknown')] += 1

    all_types = sorted({t for d in by_type_split.values() for t in d})
    header    = f"  {'Type':25s}" + "".join(f"  {s:6s}" for s in splits)
    print(header)
    for t in all_types:
        row = f"  {t:25s}"
        for s in splits:
            row += f"  {by_type_split[s].get(t, 0):6d}"
        print(row)
    print()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"  Splits saved → {output_path}\n")

    return splits


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create dataset splits from manifest')
    parser.add_argument('--manifest', type=str, default='data/processed/manifest.json')
    parser.add_argument('--output',   type=str, default='data/processed/splits.json')
    parser.add_argument('--train',    type=float, default=0.75)
    parser.add_argument('--val',      type=float, default=0.15)
    parser.add_argument('--test',     type=float, default=0.10)
    parser.add_argument('--seed',     type=int,   default=42)
    args = parser.parse_args()

    make_splits(
        manifest_path = args.manifest,
        output_path   = args.output,
        ratios        = (args.train, args.val, args.test),
        seed          = args.seed,
    )