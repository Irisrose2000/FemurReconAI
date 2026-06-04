"""
models/losses.py — Loss functions for femur segmentation & bone completion.

Segmentation
    → BinaryDiceLoss + BCEWithLogitsLoss  (standard medical segmentation combo)

Bone Completion
    → CompletionLoss = λ_bce·BCE + λ_dice·Dice + λ_focal·Focal + λ_surf·Surface

Surface Consistency Loss
    Penalises voxels at the bone surface more heavily — forces the network
    to get crisp cortical shell boundaries rather than blurry interior fill.
    Implemented with a lightweight 3-D Laplacian edge detector applied to the
    ground-truth mask; edges get ×(1 + weight) in the loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

EPS = 1e-6


def _sigmoid_if_logits(x: torch.Tensor, from_logits: bool) -> torch.Tensor:
    return torch.sigmoid(x) if from_logits else x


# ═══════════════════════════════════════════════════════════════════════════
# Dice Loss
# ═══════════════════════════════════════════════════════════════════════════

class BinaryDiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

    Works with probability maps (after sigmoid) or raw logits.
    Smooth = 1 prevents division by zero on empty volumes.
    """

    def __init__(self, smooth: float = 1.0, from_logits: bool = True):
        super().__init__()
        self.smooth      = smooth
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = _sigmoid_if_logits(pred, self.from_logits).view(-1)
        t = target.float().view(-1)

        intersection = (p * t).sum()
        dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return 1.0 - dice


# ═══════════════════════════════════════════════════════════════════════════
# Focal Loss
# ═══════════════════════════════════════════════════════════════════════════

class BinaryFocalLoss(nn.Module):
    """
    Focal loss for imbalanced bone / background classes.

    γ=2  down-weights easy background voxels;
    α    balances positive (bone) class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, from_logits: bool = True):
        super().__init__()
        self.gamma       = gamma
        self.alpha       = alpha
        self.from_logits = from_logits

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(pred, target.float(), reduction="none")
            p   = torch.sigmoid(pred)
        else:
            bce = F.binary_cross_entropy(pred, target.float(), reduction="none")
            p   = pred

        p_t   = target * p + (1 - target) * (1 - p)
        alpha = target * self.alpha + (1 - target) * (1 - self.alpha)
        fl    = alpha * ((1 - p_t) ** self.gamma) * bce
        return fl.mean()


# ═══════════════════════════════════════════════════════════════════════════
# Surface Consistency Loss
# ═══════════════════════════════════════════════════════════════════════════

def _laplacian3d_kernel() -> torch.Tensor:
    """
    3-D discrete Laplacian kernel (26-connectivity).
    Detects voxels that are on the boundary between bone and background.
    """
    k = -torch.ones(3, 3, 3)
    k[1, 1, 1] = 26.0
    return k.view(1, 1, 3, 3, 3)


class SurfaceConsistencyLoss(nn.Module):
    """
    Penalise errors on the bone surface more heavily than interior errors.

    Algorithm
    ---------
    1. Apply a 3-D Laplacian to the ground-truth mask  →  surface voxel map
    2. Build a weight map: w = 1 + surface_weight × (|Lap| > threshold)
    3. Compute pixel-wise BCE and multiply by weight map
    4. Return mean weighted BCE

    This forces the network to nail the cortical shell boundary, which is
    critical for accurate IM rod canal measurement.
    """

    def __init__(
        self,
        surface_weight:  float = 5.0,
        lap_threshold:   float = 0.1,
        from_logits:     bool  = True,
    ):
        super().__init__()
        self.surface_weight = surface_weight
        self.lap_threshold  = lap_threshold
        self.from_logits    = from_logits
        self.register_buffer("_kernel", _laplacian3d_kernel())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        t = target.float()

        # Surface detection via Laplacian on target
        kernel = self._kernel.to(t.device, t.dtype)
        lap    = F.conv3d(t, kernel, padding=1).abs()
        surface_mask = (lap > self.lap_threshold).float()

        # Weight map
        weight = 1.0 + self.surface_weight * surface_mask

        # Weighted BCE
        if self.from_logits:
            bce = F.binary_cross_entropy_with_logits(pred, t, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, t, reduction="none")

        return (bce * weight).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Combined Segmentation Loss
# ═══════════════════════════════════════════════════════════════════════════

class SegmentationLoss(nn.Module):
    """
    BCE + Dice for U-Net segmentation training.

    Loss = λ_bce * BCE + λ_dice * Dice
    """

    def __init__(
        self,
        lambda_bce:  float = 1.0,
        lambda_dice: float = 2.0,
    ):
        super().__init__()
        self.lambda_bce  = lambda_bce
        self.lambda_dice = lambda_dice
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice = BinaryDiceLoss(from_logits=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        l_bce  = self.bce(pred, target.float())
        l_dice = self.dice(pred, target)
        total  = self.lambda_bce * l_bce + self.lambda_dice * l_dice
        return {
            "total": total,
            "bce":   l_bce.detach(),
            "dice":  l_dice.detach(),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Combined Completion Loss
# ═══════════════════════════════════════════════════════════════════════════

class CompletionLoss(nn.Module):
    """
    Multi-component loss for bone completion training.

    Loss = λ_bce  * BCE
         + λ_dice * Dice
         + λ_focal* Focal
         + λ_surf * SurfaceConsistency

    Returns a dict so the trainer can log each component separately.
    """

    def __init__(
        self,
        lambda_bce:     float = 1.0,
        lambda_dice:    float = 2.0,
        lambda_focal:   float = 0.5,
        lambda_surface: float = 0.3,
    ):
        super().__init__()
        self.lw = dict(
            bce     = lambda_bce,
            dice    = lambda_dice,
            focal   = lambda_focal,
            surface = lambda_surface,
        )
        self.bce     = nn.BCEWithLogitsLoss()
        self.dice    = BinaryDiceLoss(from_logits=True)
        self.focal   = BinaryFocalLoss(from_logits=True)
        self.surface = SurfaceConsistencyLoss(from_logits=True)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        losses = {
            "bce":     self.bce(pred, target.float()),
            "dice":    self.dice(pred, target),
            "focal":   self.focal(pred, target),
            "surface": self.surface(pred, target),
        }
        total = sum(self.lw[k] * v for k, v in losses.items())
        return {
            "total": total,
            **{k: v.detach() for k, v in losses.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════
# Metric helpers (not losses but used in training loops)
# ═══════════════════════════════════════════════════════════════════════════

def dice_score(pred_binary: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    """Compute Dice coefficient between two binary tensors (numpy-friendly)."""
    p = pred_binary.float().view(-1)
    t = target.float().view(-1)
    inter = (p * t).sum()
    return ((2 * inter + smooth) / (p.sum() + t.sum() + smooth)).item()


def iou_score(pred_binary: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> float:
    p = pred_binary.float().view(-1)
    t = target.float().view(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def volume_error_percent(pred_binary: torch.Tensor, target: torch.Tensor) -> float:
    """Percentage volume error: |V_pred - V_gt| / V_gt × 100."""
    vp = pred_binary.float().sum().item()
    vt = target.float().sum().item()
    if vt < EPS:
        return 0.0
    return abs(vp - vt) / vt * 100.0