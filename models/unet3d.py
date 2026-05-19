"""
models/unet3d.py — 3-D U-Net for femur segmentation.

Architecture
------------
    Input : (B, 1,  D, H, W)   — normalised CT volume
    Output: (B, 1,  D, H, W)   — sigmoid probability map (bone = 1)

Encoder: 4 down-sampling levels, each = (Conv→BN→ReLU) × 2
Bottleneck: ResBlock with dropout
Decoder: 4 up-sampling levels, each = UpSample + skip + (Conv→BN→ReLU) × 2
Head: 1×1×1 conv → sigmoid

All convolutions are 3×3×3 with padding=1 (same spatial size).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBnRelu blocks."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBnRelu(in_ch, out_ch, dropout),
            ConvBnRelu(out_ch, out_ch, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResBlock3D(nn.Module):
    """Residual block with a skip projection if channels differ."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.bn   = nn.BatchNorm3d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x) + self.bn(self.skip(x)), inplace=True)


class DownBlock(nn.Module):
    """MaxPool3d → DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.pool  = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv  = ResBlock3D(in_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Trilinear upsample → concat skip → DoubleConv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = ResBlock3D(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle odd-sized inputs
        x = F.pad(x, _pad_diff(x, skip))
        return self.conv(torch.cat([x, skip], dim=1))


def _pad_diff(x: torch.Tensor, ref: torch.Tensor) -> List[int]:
    """Compute F.pad argument so x matches ref spatial size."""
    diffs = [ref.shape[i] - x.shape[i] for i in (4, 3, 2)]
    padding = []
    for d in diffs:
        padding += [d // 2, d - d // 2]
    return padding


# ═══════════════════════════════════════════════════════════════════════════
# U-Net
# ═══════════════════════════════════════════════════════════════════════════

class UNet3D(nn.Module):
    """
    Parameters
    ----------
    in_channels  : number of input channels (1 for mono CT)
    out_channels : 1 for binary segmentation; >1 for multi-class
    base_filters : width of first encoder level; doubles each level
    depth        : number of encoder / decoder levels (default 4)
    dropout      : spatial dropout applied inside every ResBlock
    """

    def __init__(
        self,
        in_channels: int  = 1,
        out_channels: int = 1,
        base_filters: int = 32,
        depth: int        = 4,
        dropout: float    = 0.10,
    ):
        super().__init__()
        f = base_filters

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc_in = ResBlock3D(in_channels, f, dropout)          # f

        self.down_blocks: nn.ModuleList = nn.ModuleList()
        enc_channels = [f]
        for i in range(depth - 1):
            in_f, out_f = f * (2 ** i), f * (2 ** (i + 1))
            self.down_blocks.append(DownBlock(in_f, out_f, dropout))
            enc_channels.append(out_f)

        # ── Bottleneck ────────────────────────────────────────────────────
        bn_in  = f * (2 ** (depth - 1))
        bn_out = bn_in * 2
        self.bottleneck = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResBlock3D(bn_in, bn_out, dropout),
        )

        # ── Decoder ──────────────────────────────────────────────────────
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        in_f = bn_out
        for i in range(depth - 1, -1, -1):
            skip_f = enc_channels[i]
            out_f  = f * (2 ** i) if i > 0 else f
            self.up_blocks.append(UpBlock(in_f, skip_f, out_f, dropout))
            in_f = out_f

        # ── Output head ───────────────────────────────────────────────────
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skips = []
        x = self.enc_in(x)
        skips.append(x)
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, block in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            x = block(x, skip)

        return self.head(x)          # raw logits; apply sigmoid outside

    # ── Convenience ───────────────────────────────────────────────────────
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary mask. x should be a single-item batch on the right device."""
        with torch.no_grad():
            logits = self.forward(x)
            probs  = torch.sigmoid(logits)
            return (probs > threshold).float()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# Quick smoke-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=1, base_filters=16, depth=4)
    print(f"Parameters: {model.count_parameters():,}")

    dummy = torch.zeros(1, 1, 64, 64, 32)
    out   = model(dummy)
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")
    assert out.shape == dummy.shape, "Shape mismatch!"
    print("✓ UNet3D OK")