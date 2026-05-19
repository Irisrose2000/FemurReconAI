"""
models/completion_net.py — Bone Completion Network.

Takes a FRACTURED femur binary mask and predicts the COMPLETE (intact) femur.

Architecture
------------
    Encoder  : 4-level 3-D ResNet encoder  →  latent feature map
    Bottleneck: Self-attention block (channel + spatial)
    Decoder  : 4-level decoder with U-Net skip connections
    Head     : 1×1×1 conv → sigmoid probability

Why attention at the bottleneck?
    Fractures break the femur into disconnected fragments. Pure convolutions
    struggle to "bridge the gap". Channel-wise self-attention lets the network
    reason about global shape continuity across the volume.

Input  : (B, 1, D, H, W)  — fractured binary mask (float 0/1)
Output : (B, 1, D, H, W)  — completed bone probability map [0,1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Shared building blocks
# ═══════════════════════════════════════════════════════════════════════════

class ConvBnRelu3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock3D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu3D(channels, channels, dropout),
            ConvBnRelu3D(channels, channels, 0.0),
        )
        self.bn = nn.BatchNorm3d(channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.block(x)) + x, 0.2, inplace=True)


# ═══════════════════════════════════════════════════════════════════════════
# Attention modules
# ═══════════════════════════════════════════════════════════════════════════

class ChannelAttention3D(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Lets the network weight which feature channels (shape primitives)
    are most relevant for completing the broken bone.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1, 1)
        return x * w


class SpatialAttention3D(nn.Module):
    """
    Cross-axis spatial attention: highlights the region of the fracture gap
    so the decoder focuses reconstruction effort there.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        w   = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM3D(nn.Module):
    """Channel + Spatial attention (CBAM) applied sequentially."""
    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention3D(channels)
        self.sa = SpatialAttention3D()

    def forward(self, x):
        return self.sa(self.ca(x))


# ═══════════════════════════════════════════════════════════════════════════
# Encoder / Decoder blocks
# ═══════════════════════════════════════════════════════════════════════════

class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu3D(in_ch, out_ch, dropout),
            ResBlock3D(out_ch, dropout),
        )
        self.down = nn.MaxPool3d(2, 2)

    def forward(self, x):
        skip = self.conv(x)    # save before pooling
        return self.down(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Sequential(
            ConvBnRelu3D(in_ch + skip_ch, out_ch, dropout),
            ResBlock3D(out_ch, dropout),
        )
        self.attn = CBAM3D(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if spatial dims differ (odd input sizes)
        diff = [skip.shape[i] - x.shape[i] for i in (2, 3, 4)]
        x = F.pad(x, [
            diff[2] // 2, diff[2] - diff[2] // 2,
            diff[1] // 2, diff[1] - diff[1] // 2,
            diff[0] // 2, diff[0] - diff[0] // 2,
        ])
        x = torch.cat([x, skip], dim=1)
        return self.attn(self.conv(x))


# ═══════════════════════════════════════════════════════════════════════════
# Bottleneck with multi-resolution context
# ═══════════════════════════════════════════════════════════════════════════

class AtrousBottleneck(nn.Module):
    """
    Atrous Spatial Pyramid Pooling (ASPP) adapted to 3-D.

    Captures multi-scale context:
      - rate 1  → local bone surface detail
      - rate 2  → cortical shell continuity
      - rate 4  → global femoral shape / axis

    Critical for spanning the fracture gap — a standard conv sees only
    a few voxels; dilated convs see across the entire gap at once.
    """
    def __init__(self, channels: int):
        super().__init__()
        mid = channels // 4
        self.branches = nn.ModuleList([
            nn.Conv3d(channels, mid, 1, bias=False),                          # 1×1
            nn.Conv3d(channels, mid, 3, padding=1,  dilation=1,  bias=False), # r=1
            nn.Conv3d(channels, mid, 3, padding=2,  dilation=2,  bias=False), # r=2
            nn.Conv3d(channels, mid, 3, padding=4,  dilation=4,  bias=False), # r=4
        ])
        self.fuse = nn.Sequential(
            nn.Conv3d(mid * 4, channels, 1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.cbam = CBAM3D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([b(x) for b in self.branches], dim=1)
        return self.cbam(self.fuse(feats)) + x   # residual


# ═══════════════════════════════════════════════════════════════════════════
# Full Completion Network
# ═══════════════════════════════════════════════════════════════════════════

class BoneCompletionNet(nn.Module):
    """
    Parameters
    ----------
    in_channels  : 1 (binary fractured mask)
    out_channels : 1 (completed bone probability map)
    base_filters : feature width at the first encoder level
    depth        : encoder / decoder levels  (default 4)
    dropout      : spatial dropout inside residual blocks
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        out_channels: int   = 1,
        base_filters: int   = 32,
        depth:        int   = 4,
        dropout:      float = 0.10,
    ):
        super().__init__()
        f = base_filters

        # ── Stem ─────────────────────────────────────────────────────────
        self.stem = ConvBnRelu3D(in_channels, f)

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        enc_channels  = []
        in_f = f
        for i in range(depth):
            out_f = f * (2 ** i)
            self.encoders.append(EncoderBlock(in_f, out_f, dropout))
            enc_channels.append(out_f)
            in_f = out_f

        # ── Bottleneck ────────────────────────────────────────────────────
        btn_ch = f * (2 ** depth)
        self.bottleneck_down = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ConvBnRelu3D(in_f, btn_ch, dropout),
            ResBlock3D(btn_ch, dropout),
        )
        self.aspp = AtrousBottleneck(btn_ch)

        # ── Decoder ──────────────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        in_f = btn_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_channels[i]
            out_f   = f * (2 ** i) if i > 0 else f
            self.decoders.append(DecoderBlock(in_f, skip_ch, out_f, dropout))
            in_f = out_f

        # ── Head ─────────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv3d(f, f // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(f // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(f // 2, out_channels, kernel_size=1),
        )

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, D, H, W) fractured binary mask
        Returns raw logits; apply sigmoid for probabilities.
        """
        x = self.stem(x)

        # Encode
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck_down(x)
        x = self.aspp(x)

        # Decode (reverse skip order)
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            x = dec(x, skip)

        return self.head(x)

    # ── Inference helper ─────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, fractured_mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Parameters
        ----------
        fractured_mask : (1, 1, D, H, W) float32 tensor on the correct device
        threshold      : sigmoid threshold for binarisation

        Returns
        -------
        completed_mask : (1, 1, D, H, W) binary float32 tensor
        """
        logits = self.forward(fractured_mask)
        return (torch.sigmoid(logits) > threshold).float()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════
# Smoke-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = BoneCompletionNet(base_filters=16, depth=4)
    print(f"Parameters: {model.count_parameters():,}")

    x   = torch.zeros(1, 1, 64, 64, 32)
    out = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    assert out.shape == x.shape
    print("✓ BoneCompletionNet OK")