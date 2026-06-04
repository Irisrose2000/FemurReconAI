"""
data/dicom_loader.py — Load CT scans from DICOM / NIfTI / raw slices.

Produces a normalised 3-D numpy array (float32) in Hounsfield Units
plus the voxel spacing (mm) so downstream code stays metric-aware.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def load_ct_scan(path: str | Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a CT scan from *path*.

    Accepted inputs
    ---------------
    * A **directory** containing DICOM slices (.dcm)
    * A single **.dcm** file  (single-frame, rarely useful but supported)
    * A **.nii** or **.nii.gz** NIfTI file
    * A **.mha** / **.mhd** MetaImage file (SimpleITK)

    Returns
    -------
    volume : np.ndarray, shape (D, H, W), dtype float32
        Voxel intensities in Hounsfield Units (HU).
    spacing : (sz, sy, sx) tuple of floats
        Physical voxel size in millimetres, z first.
    """
    path = Path(path)

    if path.is_dir():
        return _load_dicom_series(path)

    suffix = "".join(path.suffixes).lower()
    if suffix in {".dcm"}:
        return _load_single_dicom(path)
    if suffix in {".nii", ".nii.gz"}:
        return _load_nifti(path)
    if suffix in {".mha", ".mhd"}:
        return _load_sitk(path)

    raise ValueError(f"Unsupported file type: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# DICOM
# ═══════════════════════════════════════════════════════════════════════════

def _load_dicom_series(directory: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a folder of DICOM slices and stack them into a volume."""
    dcm_files = sorted(directory.glob("*.dcm"))
    if not dcm_files:
        # Some scanners omit the extension
        dcm_files = sorted(
            [f for f in directory.iterdir()
             if f.is_file() and _is_dicom(f)]
        )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in {directory}")

    slices = [pydicom.dcmread(str(f)) for f in dcm_files]

    # Sort slices by Image Position (z-coordinate) for correct ordering
    try:
        slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key=lambda s: int(s.InstanceNumber))

    # ── Stack pixel arrays ────────────────────────────────────────────────
    pixel_arrays = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)
        # Apply rescale slope / intercept to get HU
        slope = float(getattr(s, "RescaleSlope", 1))
        intercept = float(getattr(s, "RescaleIntercept", 0))
        arr = arr * slope + intercept
        pixel_arrays.append(arr)

    volume = np.stack(pixel_arrays, axis=0)   # (D, H, W)

    # ── Voxel spacing ─────────────────────────────────────────────────────
    ref = slices[0]
    try:
        px, py = [float(x) for x in ref.PixelSpacing]
    except AttributeError:
        px, py = 1.0, 1.0

    if len(slices) > 1:
        try:
            z0 = float(slices[0].ImagePositionPatient[2])
            z1 = float(slices[1].ImagePositionPatient[2])
            pz = abs(z1 - z0)
        except AttributeError:
            pz = float(getattr(ref, "SliceThickness", 1.0))
    else:
        pz = float(getattr(ref, "SliceThickness", 1.0))

    return volume.astype(np.float32), (pz, py, px)


def _load_single_dicom(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept
    px, py = [float(x) for x in getattr(ds, "PixelSpacing", [1.0, 1.0])]
    pz = float(getattr(ds, "SliceThickness", 1.0))
    # single slice → add depth dimension
    return arr[np.newaxis].astype(np.float32), (pz, py, px)


def _is_dicom(path: Path) -> bool:
    """Peek at first 132 bytes for the DICM magic string."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# NIfTI
# ═══════════════════════════════════════════════════════════════════════════

def _load_nifti(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)

    # NIfTI is stored (W, H, D) — transpose to (D, H, W)
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)
    elif data.ndim == 4:
        data = data[..., 0].transpose(2, 1, 0)

    zooms = img.header.get_zooms()   # (sx, sy, sz)
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return data, spacing


# ═══════════════════════════════════════════════════════════════════════════
# SimpleITK (.mha / .mhd)
# ═══════════════════════════════════════════════════════════════════════════

def _load_sitk(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img).astype(np.float32)  # (D, H, W)
    sp = img.GetSpacing()   # (sx, sy, sz)
    return data, (float(sp[2]), float(sp[1]), float(sp[0]))