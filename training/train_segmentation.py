"""
training/train_segmentation.py — U-Net femur segmentation training loop.

Usage
-----
    python -m training.train_segmentation --data_dir data/processed --epochs 150
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import cfg
from data.dataset import SegmentationDataset
from models.unet3d import UNet3D
from models.losses import SegmentationLoss, dice_score, iou_score


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


# ═══════════════════════════════════════════════════════════════════════════
# One epoch
# ═══════════════════════════════════════════════════════════════════════════

def run_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  SegmentationLoss,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    is_train:   bool,
) -> dict:
    model.train(is_train)
    totals = dict(loss=0.0, bce=0.0, dice_loss=0.0, dice_score=0.0, iou=0.0)
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, desc="train" if is_train else "val ", leave=False):
            volume = batch["volume"].to(device)   # (B, 1, D, H, W)
            mask   = batch["mask"].to(device)     # (B, 1, D, H, W)

            logits = model(volume)
            losses = criterion(logits, mask)

            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Metrics
            with torch.no_grad():
                pred_bin = (torch.sigmoid(logits) > 0.5).float()
                ds = dice_score(pred_bin, mask)
                io = iou_score(pred_bin, mask)

            totals["loss"]       += losses["total"].item()
            totals["bce"]        += losses["bce"].item()
            totals["dice_loss"]  += losses["dice"].item()
            totals["dice_score"] += ds
            totals["iou"]        += io
            n_batches            += 1

    return {k: v / n_batches for k, v in totals.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"Device: {device}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = SegmentationDataset(args.data_dir, split="train", augment=True)
    val_ds   = SegmentationDataset(args.data_dir, split="val",   augment=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.seg_batch_size,
        shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1,
        shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels  = cfg.model.seg_in_channels,
        out_channels = cfg.model.seg_out_channels,
        base_filters = cfg.model.base_filters,
        depth        = cfg.model.seg_depth,
        dropout      = cfg.model.dropout,
    ).to(device)
    print(f"UNet3D parameters: {model.count_parameters():,}")

    # ── Optimiser & scheduler ─────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg.train.seg_lr,
        weight_decay = cfg.train.seg_weight_decay,
    )
    epochs = args.epochs or cfg.train.seg_epochs

    if cfg.train.seg_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    else:
        scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = SegmentationLoss(lambda_bce=1.0, lambda_dice=2.0)

    # ── Logging ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(cfg.train.log_dir / "segmentation"))
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice  = 0.0
    history    = []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        else:
            scheduler.step(val_metrics["loss"])

        elapsed = time.time() - t0

        # ── Logging ──────────────────────────────────────────────────────
        for tag, m in [("train", train_metrics), ("val", val_metrics)]:
            for k, v in m.items():
                writer.add_scalar(f"seg/{tag}/{k}", v, epoch)
        writer.add_scalar("seg/lr", optimizer.param_groups[0]["lr"], epoch)

        log = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "val_loss":   round(val_metrics["loss"],   4),
            "val_dice":   round(val_metrics["dice_score"], 4),
            "val_iou":    round(val_metrics["iou"],    4),
            "lr":         optimizer.param_groups[0]["lr"],
            "time_s":     round(elapsed, 1),
        }
        history.append(log)
        print(
            f"[Seg] Ep {epoch:03d}/{epochs} | "
            f"loss {log['train_loss']:.4f} → {log['val_loss']:.4f} | "
            f"dice {log['val_dice']:.4f} | iou {log['val_iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoints ──────────────────────────────────────────────────
        state = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "val_dice":   log["val_dice"],
            "model_cfg":  {
                "in_channels":  cfg.model.seg_in_channels,
                "out_channels": cfg.model.seg_out_channels,
                "base_filters": cfg.model.base_filters,
                "depth":        cfg.model.seg_depth,
                "dropout":      cfg.model.dropout,
            },
        }

        if log["val_dice"] > best_dice:
            best_dice = log["val_dice"]
            save_checkpoint(state, ckpt_dir / "seg_best.pth")
            print(f"  ↑ New best Dice: {best_dice:.4f} — checkpoint saved")

        if epoch % cfg.train.save_every == 0:
            save_checkpoint(state, ckpt_dir / f"seg_ep{epoch:04d}.pth")

    # ── Save history ──────────────────────────────────────────────────────
    with open(ckpt_dir / "seg_history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train femur segmentation U-Net")
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Directory of preprocessed .npz files")
    parser.add_argument("--epochs",   type=int, default=None,
                        help="Override epoch count from config")
    args = parser.parse_args()
    train(args)