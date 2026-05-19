"""
config.py — Central configuration for Femur Reconstruction ML System
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class DataConfig:
    # ── Voxel grid size (D x H x W). Femur is tall → more depth slices
    volume_shape: Tuple[int, int, int] = (128, 128, 64)

    # ── Hounsfield Unit (HU) window for bone isolation
    hu_min: float = 200.0      # cortical bone starts ~400, cancellous ~200
    hu_max: float = 1800.0     # above 1800 = metal artefacts / teeth

    # ── Target voxel spacing after resampling (mm)
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.5)

    # ── Paths
    raw_data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    splits_file: Path = Path("data/splits.json")


@dataclass
class ModelConfig:
    # ── Shared
    base_filters: int = 32       # scales the width of both models

    # ── Segmentation U-Net
    seg_in_channels: int = 1
    seg_out_channels: int = 1    # binary: femur vs background
    seg_depth: int = 4           # encoder / decoder levels

    # ── Bone Completion Network
    comp_in_channels: int = 1    # fractured binary mask
    comp_out_channels: int = 1   # completed binary mask
    comp_latent_dim: int = 512
    comp_use_skip: bool = True   # U-Net style skip connections

    # ── Dropout for regularisation
    dropout: float = 0.10


@dataclass
class TrainConfig:
    # ── General
    seed: int = 42
    device: str = "cpu"         # "cpu" if no GPU
    num_workers: int = 0

    # ── Segmentation
    seg_epochs: int = 150
    seg_batch_size: int = 1
    seg_lr: float = 1e-4
    seg_weight_decay: float = 1e-5
    seg_scheduler: str = "cosine"   # "cosine" | "step" | "plateau"

    # ── Bone Completion
    comp_epochs: int = 200
    comp_batch_size: int = 1
    comp_lr: float = 1e-4
    comp_weight_decay: float = 1e-5

    # ── Loss weights (completion)
    lambda_bce: float = 1.0
    lambda_dice: float = 2.0
    lambda_focal: float = 0.5
    lambda_surface: float = 0.3   # surface-consistency term

    # ── Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 10          # save checkpoint every N epochs
    log_dir: Path = Path("runs")


@dataclass
class InferenceConfig:
    seg_checkpoint: Path = Path("checkpoints/seg_best.pth")
    comp_checkpoint: Path = Path("checkpoints/comp_best.pth")
    output_dir: Path = Path("results")
    # Marching-cubes iso-level for mesh extraction
    iso_level: float = 0.5
    # Minimum fragment volume (mm³) — smaller = noise
    min_fragment_volume_mm3: float = 500.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# ── Singleton ──────────────────────────────────────────────────────────────
cfg = Config()