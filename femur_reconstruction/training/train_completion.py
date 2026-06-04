"""
training/train_completion.py — Bone completion network training loop.

Trains BoneCompletionNet to reconstruct an intact femur from a fractured mask.
Training pairs are generated on-the-fly by SyntheticFractureAugmenter.

Usage
-----
    python -m training.train_completion --data_dir data/processed --epochs 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import cfg
from data.dataset import BoneCompletionDataset
from data.preprocessor import SyntheticFractureAugmenter
from models.completion_net import BoneCompletionNet
from models.losses import CompletionLoss, dice_score, iou_score, volume_error_percent


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
    criterion:  CompletionLoss,
    optimizer:  torch.optim.Optimizer | None,
    device:     torch.device,
    is_train:   bool,
) -> dict:
    model.train(is_train)
    totals = dict(
        loss=0.0, bce=0.0, dice_loss=0.0, focal=0.0, surface=0.0,
        dice_score=0.0, iou=0.0, vol_err=0.0,
    )
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for batch in tqdm(loader, desc="train" if is_train else "val ", leave=False):
            fractured = batch["fractured"].to(device)   # (B, 1, D, H, W)
            complete  = batch["complete"].to(device)    # (B, 1, D, H, W)

            logits = model(fractured)
            losses = criterion(logits, complete)

            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Metrics
            with torch.no_grad():
                pred_bin = (torch.sigmoid(logits) > 0.5).float()
                ds  = dice_score(pred_bin, complete)
                io  = iou_score(pred_bin, complete)
                ve  = volume_error_percent(pred_bin, complete)

            totals["loss"]       += losses["total"].item()
            totals["bce"]        += losses["bce"].item()
            totals["dice_loss"]  += losses["dice"].item()
            totals["focal"]      += losses["focal"].item()
            totals["surface"]    += losses["surface"].item()
            totals["dice_score"] += ds
            totals["iou"]        += io
            totals["vol_err"]    += ve
            n_batches            += 1

    return {k: v / n_batches for k, v in totals.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Curriculum: increase fracture severity as training progresses
# ═══════════════════════════════════════════════════════════════════════════

def make_fracture_augmenter(epoch: int, total_epochs: int) -> SyntheticFractureAugmenter:
    """
    Curriculum learning: start with simple transverse fractures and small gaps,
    gradually increase gap size and comminution probability.
    """
    progress = epoch / total_epochs          # 0 → 1
    gap_min  = 2.0 + progress * 3.0         # 2 → 5 mm
    gap_max  = 8.0 + progress * 12.0        # 8 → 20 mm
    comm_p   = 0.10 + progress * 0.40       # 10 → 50 %
    return SyntheticFractureAugmenter(
        gap_range_mm       = (gap_min, gap_max),
        comminution_prob   = comm_p,
        max_fragments      = 3,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    print(f"Device: {device}")

    epochs = args.epochs or cfg.train.comp_epochs

    # ── Model ─────────────────────────────────────────────────────────────
    model = BoneCompletionNet(
        in_channels  = cfg.model.comp_in_channels,
        out_channels = cfg.model.comp_out_channels,
        base_filters = cfg.model.base_filters,
        depth        = 4,
        dropout      = cfg.model.dropout,
    ).to(device)
    print(f"BoneCompletionNet parameters: {model.count_parameters():,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg.train.comp_lr,
        weight_decay = cfg.train.comp_weight_decay,
    )
    # Warm restarts — useful with curriculum: resets LR when fracture severity jumps
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-7)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CompletionLoss(
        lambda_bce     = cfg.train.lambda_bce,
        lambda_dice    = cfg.train.lambda_dice,
        lambda_focal   = cfg.train.lambda_focal,
        lambda_surface = cfg.train.lambda_surface,
    )

    # ── Logging ───────────────────────────────────────────────────────────
    writer   = SummaryWriter(log_dir=str(cfg.train.log_dir / "completion"))
    ckpt_dir = cfg.train.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0.0
    history   = []

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Rebuild datasets with curriculum-appropriate fracture augmenter
        aug = make_fracture_augmenter(epoch, epochs)
        train_ds = BoneCompletionDataset(
            args.data_dir, split="train", augment=True, fracture_augmenter=aug,
        )
        val_ds = BoneCompletionDataset(
            args.data_dir, split="val", augment=False,
            fracture_augmenter=SyntheticFractureAugmenter(gap_range_mm=(5.0, 15.0)),
        )

        train_loader = DataLoader(
            train_ds, batch_size=cfg.train.comp_batch_size,
            shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1,
            shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True,
        )

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_m   = run_epoch(model, val_loader,   criterion, None,      device, False)

        scheduler.step(epoch)
        elapsed = time.time() - t0

        # ── TensorBoard ──────────────────────────────────────────────────
        for tag, m in [("train", train_m), ("val", val_m)]:
            for k, v in m.items():
                writer.add_scalar(f"comp/{tag}/{k}", v, epoch)
        writer.add_scalar("comp/lr", optimizer.param_groups[0]["lr"], epoch)

        log = {
            "epoch":       epoch,
            "train_loss":  round(train_m["loss"],       4),
            "val_loss":    round(val_m["loss"],         4),
            "val_dice":    round(val_m["dice_score"],   4),
            "val_iou":     round(val_m["iou"],          4),
            "val_vol_err": round(val_m["vol_err"],      2),
            "lr":          optimizer.param_groups[0]["lr"],
            "time_s":      round(elapsed, 1),
        }
        history.append(log)

        print(
            f"[Comp] Ep {epoch:03d}/{epochs} | "
            f"loss {log['train_loss']:.4f}→{log['val_loss']:.4f} | "
            f"dice {log['val_dice']:.4f} | vol_err {log['val_vol_err']:.1f}% | "
            f"{elapsed:.1f}s"
        )

        # ── Checkpoints ──────────────────────────────────────────────────
        state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_dice":  log["val_dice"],
        }

        if log["val_dice"] > best_dice:
            best_dice = log["val_dice"]
            save_checkpoint(state, ckpt_dir / "comp_best.pth")
            print(f"  ↑ New best Dice: {best_dice:.4f} — checkpoint saved")

        if epoch % cfg.train.save_every == 0:
            save_checkpoint(state, ckpt_dir / f"comp_ep{epoch:04d}.pth")

    with open(ckpt_dir / "comp_history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bone completion network")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--epochs",   type=int, default=None)
    args = parser.parse_args()
    train(args)