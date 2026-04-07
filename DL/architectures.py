"""
architectures.py
================
Deep-learning model architectures for **binary semantic segmentation** of
raster data in QGIS.

Supported square input sizes (height == width):
    64, 128, 256, 512, 1024  pixels

Constraints
-----------
* Input patches must be **square** (H == W).
* Maximum patch size: 1024 × 1024 px.
* Supported sizes listed in ``SUPPORTED_SIZES`` – anything outside that list
  will raise an AssertionError; the plugin should show ``UNSUPPORTED_SIZE_WARNING``
  to the user before the model is constructed.
* Input channels: 1 – 10  (``MAX_IN_CHANNELS``).
* Output channels: **1** (binary segmentation – raw logit; apply ``torch.sigmoid``
  externally to get a probability map, then threshold at 0.5).

Models
------
* ``UNet``          – classic encoder-decoder with skip connections
* ``AttentionUNet`` – UNet with soft attention gates on skip connections
* ``UNetPP``        – UNet++ with dense nested skip connections
* ``SwinUNet``      – pure-transformer Swin-UNet encoder-decoder
* ``LinkNet``       – lightweight residual encoder-decoder (element-wise skip adds)
* ``DeepLabV3Plus`` – dilated CNN encoder with ASPP + lightweight decoder
* ``SegFormer``     – hierarchical Mix Transformer encoder + all-MLP decoder

Factory
-------
Use ``build_model(name, in_channels, img_size)`` to instantiate any model by its
string key: ``"unet"``, ``"attention_unet"``, ``"unet_pp"``, ``"swin_unet"``,
``"linknet"``, ``"deeplabv3plus"``, ``"segformer"``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Public constants – used by the plugin UI for validation
# ---------------------------------------------------------------------------

SUPPORTED_SIZES: List[int] = [64, 128, 256, 512, 1024]
"""Square patch sizes (px) supported by all architectures."""

MAX_IN_CHANNELS: int = 40
"""Maximum number of input raster bands accepted by any model."""

MAX_SIZE: int = 1024
"""Hard upper limit on spatial input dimension (px)."""

UNSUPPORTED_SIZE_WARNING: str = (
    "Unsupported image patch size!\n\n"
    "Only the following square sizes (height = width) are supported:\n"
    f"  {SUPPORTED_SIZES}\n\n"
    f"The maximum allowed size is {MAX_SIZE} × {MAX_SIZE} px.\n"
    "Larger sizes and non-square patches are not accepted by any model."
)

# ===========================================================================
# Shared CNN building blocks
# ===========================================================================


class DoubleConv(nn.Module):
    """Two consecutive (Conv 3×3 → BN → ReLU) blocks."""

    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            mid_ch: Optional[int] = None) -> None:
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def _up_cat(deep: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
    """Bilinearly upsample *deep* to match *skips[0]*'s spatial size, then
    concatenate all tensors along the channel axis.

    Args:
        deep : Lower-resolution feature map to upsample.
        skips: Higher-resolution skip-connection feature maps (all same spatial size).

    Returns:
        Concatenated tensor ``[*skips, deep_upsampled]``.
    """
    deep = F.interpolate(deep,
                         size=skips[0].shape[2:],
                         mode="bilinear",
                         align_corners=True)
    return torch.cat([*skips, deep], dim=1)


# ===========================================================================
# 1. U-Net
# ===========================================================================


class UNet(nn.Module):
    """
    Standard U-Net for binary semantic segmentation.

    Architecture: 4-level encoder (MaxPool + DoubleConv) → bottleneck →
    4-level decoder (bilinear upsample + skip concat + DoubleConv) →
    1×1 Conv head (1 output channel).

    Args:
        in_channels   : Raster bands fed into the model (1 – 40).
        img_size      : Square spatial size of each input patch; must be one of
                        ``SUPPORTED_SIZES``.
        base_channels : Feature-map width at the first encoder stage (default 64).
                        Doubles at each stage: 64 → 128 → 256 → 512 → bottleneck 1024.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 256,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        assert 1 <= in_channels <= MAX_IN_CHANNELS, (
            f"in_channels must be 1 – {MAX_IN_CHANNELS}, got {in_channels}."
        )
        assert img_size in SUPPORTED_SIZES, UNSUPPORTED_SIZE_WARNING

        c = base_channels
        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 2, c * 4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 4, c * 8))
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(
                c * 8, c * 16))
        # Decoder
        self.dec4 = DoubleConv(c * 16 + c * 8, c * 8)
        self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)
        self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)
        self.dec1 = DoubleConv(c * 2 + c, c)
        # Segmentation head (raw logit)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)

        d4 = self.dec4(_up_cat(b, s4))
        d3 = self.dec3(_up_cat(d4, s3))
        d2 = self.dec2(_up_cat(d3, s2))
        d1 = self.dec1(_up_cat(d2, s1))
        return self.head(d1)


# ===========================================================================
# 2. Attention U-Net
# ===========================================================================


class _AttentionGate(nn.Module):
    """Soft spatial attention gate (Oktay et al., 2018).

    Computes a per-pixel attention coefficient α ∈ (0, 1) from a gating signal
    *g* (decoder) and a skip feature *x* (encoder), then returns *x · α*.

    Args:
        g_ch    : Channels in the gating signal (decoder feature map).
        x_ch    : Channels in the skip-connection feature map.
        inter_ch: Intermediate channel width for the gate computation.
    """

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.Wg = nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False)
        self.Wx = nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g_up = F.interpolate(g,
                             size=x.shape[2:],
                             mode="bilinear",
                             align_corners=True)
        attn = torch.sigmoid(
            self.psi(
                F.relu(
                    self.Wg(g_up) +
                    self.Wx(x),
                    inplace=True)))
        return x * attn


class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary semantic segmentation.

    Identical encoder/bottleneck/decoder to ``UNet``, with soft attention
    gates applied to every skip connection before it enters the decoder
    (Oktay et al., "Attention U-Net", 2018).

    Args:
        in_channels   : Raster bands fed into the model (1 – 40).
        img_size      : Square spatial size of each input patch; must be one of
                        ``SUPPORTED_SIZES``.
        base_channels : Feature-map width at the first encoder stage (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 256,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        assert 1 <= in_channels <= MAX_IN_CHANNELS, (
            f"in_channels must be 1 – {MAX_IN_CHANNELS}, got {in_channels}."
        )
        assert img_size in SUPPORTED_SIZES, UNSUPPORTED_SIZE_WARNING

        c = base_channels
        # Encoder
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c, c * 2))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 2, c * 4))
        self.enc4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c * 4, c * 8))
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(
                c * 8, c * 16))
        # Attention gates (gating signal = decoder feature, skip = encoder
        # feature)
        self.attn4 = _AttentionGate(c * 16, c * 8, c * 8)
        self.attn3 = _AttentionGate(c * 8, c * 4, c * 4)
        self.attn2 = _AttentionGate(c * 4, c * 2, c * 2)
        self.attn1 = _AttentionGate(c * 2, c, c)
        # Decoder
        self.dec4 = DoubleConv(c * 16 + c * 8, c * 8)
        self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)
        self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)
        self.dec1 = DoubleConv(c * 2 + c, c)
        # Segmentation head
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b = self.bottleneck(s4)

        a4 = self.attn4(b, s4)
        d4 = self.dec4(_up_cat(b, a4))
        a3 = self.attn3(d4, s3)
        d3 = self.dec3(_up_cat(d4, a3))
        a2 = self.attn2(d3, s2)
        d2 = self.dec2(_up_cat(d3, a2))
        a1 = self.attn1(d2, s1)
        d1 = self.dec1(_up_cat(d2, a1))
        return self.head(d1)


# ===========================================================================
# 3. UNet++ (Nested U-Net)
# ===========================================================================


class UNetPP(nn.Module):
    """
    UNet++ for binary semantic segmentation.

    Replaces the plain skip connections of U-Net with a grid of densely
    connected intermediate nodes X^{i,j}.  Each node receives the upsampled
    output of the node directly below it **plus** all earlier nodes at the
    same encoder depth (Zhou et al., "UNet++", 2018).

    Notation: depth index *i* = 0 (shallowest) … 4 (deepest / bottleneck).
              column index *j* = 0 (encoder) … 4 (final decoder).

    Args:
        in_channels       : Raster bands fed into the model (1 – 40).
        img_size          : Square spatial size of each input patch; must be in
                            ``SUPPORTED_SIZES``.
        base_channels     : Feature-map width at depth 0 (default 64).
                            Depths use [64, 128, 256, 512, 1024].
        deep_supervision  : When ``True``, ``forward()`` returns a *list* of
                            four raw-logit tensors (one per decoder column,
                            j = 1 … 4) instead of a single tensor.  Useful
                            during training; at inference set to ``False``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 256,
        base_channels: int = 64,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        assert 1 <= in_channels <= MAX_IN_CHANNELS, (
            f"in_channels must be 1 – {MAX_IN_CHANNELS}, got {in_channels}."
        )
        assert img_size in SUPPORTED_SIZES, UNSUPPORTED_SIZE_WARNING

        self.deep_supervision = deep_supervision
        nb = [base_channels * (2 ** i)
              for i in range(5)]  # [64,128,256,512,1024]

        # ── Encoder column (j = 0) ──────────────────────────────────────────
        self.enc = nn.ModuleList([
            # X^{0,0}
            DoubleConv(in_channels, nb[0]),
            nn.Sequential(
                nn.MaxPool2d(2), DoubleConv(
                    nb[0], nb[1])),   # X^{1,0}
            nn.Sequential(
                nn.MaxPool2d(2), DoubleConv(
                    nb[1], nb[2])),   # X^{2,0}
            nn.Sequential(
                nn.MaxPool2d(2), DoubleConv(
                    nb[2], nb[3])),   # X^{3,0}
            nn.Sequential(
                nn.MaxPool2d(2), DoubleConv(
                    nb[3], nb[4])),   # X^{4,0}
        ])

        # ── Dense decoder nodes ─────────────────────────────────────────────
        # Column j = 1
        self.node_0_1 = DoubleConv(
            nb[1] + nb[0], nb[0])  # up(X^{1,0}) + X^{0,0}
        self.node_1_1 = DoubleConv(
            nb[2] + nb[1], nb[1])  # up(X^{2,0}) + X^{1,0}
        self.node_2_1 = DoubleConv(nb[3] + nb[2], nb[2])
        self.node_3_1 = DoubleConv(nb[4] + nb[3], nb[3])

        # Column j = 2
        # up(X^{1,1}) + X^{0,0} + X^{0,1}
        self.node_0_2 = DoubleConv(nb[1] + nb[0] * 2, nb[0])
        self.node_1_2 = DoubleConv(nb[2] + nb[1] * 2, nb[1])
        self.node_2_2 = DoubleConv(nb[3] + nb[2] * 2, nb[2])

        # Column j = 3
        self.node_0_3 = DoubleConv(nb[1] + nb[0] * 3, nb[0])
        self.node_1_3 = DoubleConv(nb[2] + nb[1] * 3, nb[1])

        # Column j = 4 (final)
        self.node_0_4 = DoubleConv(nb[1] + nb[0] * 4, nb[0])

        # ── Segmentation heads ──────────────────────────────────────────────
        if self.deep_supervision:
            self.heads = nn.ModuleList(
                [nn.Conv2d(nb[0], 1, 1) for _ in range(4)])
        else:
            self.head = nn.Conv2d(nb[0], 1, 1)

    @staticmethod
    def _up_cat(deep: torch.Tensor, *skips: torch.Tensor) -> torch.Tensor:
        deep = F.interpolate(deep,
                             size=skips[0].shape[2:],
                             mode="bilinear",
                             align_corners=True)
        return torch.cat([*skips, deep], dim=1)

    def forward(self, x: torch.Tensor):
        # Encoder
        x0_0 = self.enc[0](x)
        x1_0 = self.enc[1](x0_0)
        x2_0 = self.enc[2](x1_0)
        x3_0 = self.enc[3](x2_0)
        x4_0 = self.enc[4](x3_0)

        # j = 1
        x0_1 = self.node_0_1(self._up_cat(x1_0, x0_0))
        x1_1 = self.node_1_1(self._up_cat(x2_0, x1_0))
        x2_1 = self.node_2_1(self._up_cat(x3_0, x2_0))
        x3_1 = self.node_3_1(self._up_cat(x4_0, x3_0))

        # j = 2
        x0_2 = self.node_0_2(self._up_cat(x1_1, x0_0, x0_1))
        x1_2 = self.node_1_2(self._up_cat(x2_1, x1_0, x1_1))
        x2_2 = self.node_2_2(self._up_cat(x3_1, x2_0, x2_1))

        # j = 3
        x0_3 = self.node_0_3(self._up_cat(x1_2, x0_0, x0_1, x0_2))
        x1_3 = self.node_1_3(self._up_cat(x2_2, x1_0, x1_1, x1_2))

        # j = 4
        x0_4 = self.node_0_4(self._up_cat(x1_3, x0_0, x0_1, x0_2, x0_3))

        if self.deep_supervision:
            return [
                self.heads[0](x0_1),
                self.heads[1](x0_2),
                self.heads[2](x0_3),
                self.heads[3](x0_4),
            ]
        return self.head(x0_4)


# ===========================================================================
# 4. Swin-UNet
# ===========================================================================
# Pure-PyTorch implementation – no external ViT / timm dependency.
# Reference: Cao et al., "Swin-Unet: Unet-like Pure Transformer for Medical
#            Image Segmentation", ECCVW 2022.


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    """Partition (B, H, W, C) into (B*nW, ws, ws, C) non-overlapping windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(
        windows: torch.Tensor,
        ws: int,
        H: int,
        W: int) -> torch.Tensor:
    """Invert ``_window_partition``. Returns (B, H, W, C)."""
    B = windows.shape[0] // ((H // ws) * (W // ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class _WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative-positional bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # Relative positional bias table: (2*Ws-1)^2 × num_heads
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij"))  # (2, Ws, Ws)
        # (2, Ws²)
        coords_f = coords.flatten(1)
        rel = coords_f[:, :, None] - \
            coords_f[:, None, :]          # (2, Ws², Ws²)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_idx",
                             rel.sum(-1))                # (Ws², Ws²)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        rp_bias = self.rel_pos_bias_table[self.rel_pos_idx.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, self.num_heads
        ).permute(2, 0, 1).contiguous()
        attn = attn + rp_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(torch.softmax(attn, dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class _SwinBlock(nn.Module):
    """One Swin Transformer block (W-MSA or SW-MSA followed by an MLP)."""

    def __init__(
        self,
        dim: int,
        resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        H, W = resolution
        # Clamp window size so it never exceeds the feature-map dimensions.
        self.window_size = min(window_size, H, W)
        # If the entire feature map fits in one window, shifting has no effect.
        self.shift_size = 0 if (
            self.window_size >= H and self.window_size >= W) else shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, self.window_size, num_heads,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

        # Pre-compute SW-MSA attention mask (stored as a buffer, not a
        # parameter).
        if self.shift_size > 0:
            img_mask = torch.zeros(1, H, W, 1)
            for h_sl in (slice(0, -self.window_size),
                         slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None)):
                for w_sl in (slice(0, -self.window_size),
                             slice(-self.window_size, -self.shift_size),
                             slice(-self.shift_size, None)):
                    img_mask[:, h_sl, w_sl, :] += 1  # unique region id
                    # will be overwritten below
                    img_mask[:, h_sl, w_sl, :] *= 1

            # Assign unique integer per region
            img_mask.zero_()
            cnt = 0
            for h_sl in (slice(0, -self.window_size),
                         slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None)):
                for w_sl in (slice(0, -self.window_size),
                             slice(-self.window_size, -self.shift_size),
                             slice(-self.shift_size, None)):
                    img_mask[:, h_sl, w_sl, :] = cnt
                    cnt += 1

            mask_windows = _window_partition(
                img_mask, self.window_size)    # (nW,Ws,Ws,1)
            mask_windows = mask_windows.view(-1, self.window_size ** 2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self._H = H
        self._W = W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self._H, self._W
        B, _L, C = x.shape

        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -
                           self.shift_size), dims=(1, 2))

        x_win = _window_partition(x, self.window_size).view(
            -1, self.window_size ** 2, C
        )
        x_win = self.attn(x_win, mask=self.attn_mask)
        x = _window_reverse(
            x_win.view(-1, self.window_size, self.window_size, C),
            self.window_size, H, W,
        )

        if self.shift_size > 0:
            x = torch.roll(
                x, shifts=(
                    self.shift_size, self.shift_size), dims=(
                    1, 2))

        x = shortcut + x.view(B, H * W, C)
        x = x + self.mlp(self.norm2(x))
        return x


class _PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to *embed_dim*."""

    def __init__(
            self,
            in_ch: int,
            embed_dim: int,
            patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_ch,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)           # (B, C, H/ps, W/ps)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)   # (B, Hp*Wp, C)
        return self.norm(x), Hp, Wp


class _PatchMerging(nn.Module):
    """Downsample token grid by 2× (merge 2×2 neighbouring tokens)."""

    def __init__(self, dim: int, out_dim: Optional[int] = None) -> None:
        super().__init__()
        out_dim = out_dim or dim * 2
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, H: int,
                W: int) -> Tuple[torch.Tensor, int, int]:
        B, _L, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat(
            [x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],
             x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]],
            dim=-1,
        ).view(B, -1, 4 * C)
        return self.reduction(self.norm(x)), H // 2, W // 2


class _PatchExpanding(nn.Module):
    """Upsample token grid by 2× via linear projection + pixel-shuffle."""

    def __init__(self, dim: int, out_dim: Optional[int] = None) -> None:
        super().__init__()
        out_dim = out_dim or dim // 2
        self._out = out_dim
        self.expand = nn.Linear(dim, 4 * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, H: int,
                W: int) -> Tuple[torch.Tensor, int, int]:
        B, _L, _C = x.shape
        x = self.expand(x).view(B, H, W, 2, 2, self._out)
        x = x.permute(
            0, 1, 3, 2, 4, 5).contiguous().view(
            B, H * 2 * W * 2, self._out)
        return self.norm(x), H * 2, W * 2


class _SwinStage(nn.Module):
    """A sequence of Swin Transformer blocks at one resolution."""

    def __init__(
        self,
        dim: int,
        resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock(
                dim=dim,
                resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class SwinUNet(nn.Module):
    """
    Swin-UNet for binary semantic segmentation.

    Pure-transformer encoder-decoder: patch embedding → 4 Swin Transformer
    encoder stages (with patch-merging downsampling) → 3 symmetric decoder
    stages (with patch-expanding upsampling and skip connections) →
    pixel-shuffle final upsample → 1×1 Conv head.

    Args:
        in_channels  : Raster bands fed into the model (1 – 40).
        img_size     : Square spatial size of each input patch; must be one of
                       ``SUPPORTED_SIZES``.
        patch_size   : Patch embedding stride / kernel (default 4).
        embed_dim    : Base embedding dimension (default 96).
                       Stage dims: 96 → 192 → 384 → 768.
        depths       : Swin blocks per encoder stage (default [2, 2, 6, 2]).
        num_heads    : Attention heads per stage (default [3, 6, 12, 24]).
        window_size  : Attention window size in token units (default 8).
                       Automatically clamped to the feature-map size when the
                       feature map is smaller than the requested window.

    Notes
    -----
    * For ``img_size = 64`` the bottleneck resolution is 2×2 tokens; a single
      window covers the whole map (no cyclic shift) — this is handled automatically.
    * ``forward()`` returns a single-channel raw logit tensor with the same
      spatial size as the input.  Apply ``torch.sigmoid`` to get probabilities.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 256,
        patch_size: int = 4,
        embed_dim: int = 96,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        window_size: int = 8,
    ) -> None:
        super().__init__()
        assert 1 <= in_channels <= MAX_IN_CHANNELS, (
            f"in_channels must be 1 – {MAX_IN_CHANNELS}, got {in_channels}."
        )
        assert img_size in SUPPORTED_SIZES, UNSUPPORTED_SIZE_WARNING

        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]
        assert len(depths) == len(num_heads) == 4, "Exactly 4 stages required."

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        patch_res = img_size // patch_size          # token grid side length
        # Encoder stage resolutions: patch_res, patch_res/2, patch_res/4,
        # patch_res/8
        enc_res = [(patch_res >> i, patch_res >> i) for i in range(4)]
        dims = [embed_dim * (2 ** i) for i in range(4)]   # [96, 192, 384, 768]

        # ── Patch embedding ─────────────────────────────────────────────────
        self.patch_embed = _PatchEmbed(in_channels, embed_dim, patch_size)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc_stages = nn.ModuleList([
            _SwinStage(dims[i], enc_res[i], depths[i], num_heads[i], window_size)
            for i in range(4)
        ])
        self.patch_merging = nn.ModuleList([
            _PatchMerging(dims[i], dims[i + 1]) for i in range(3)
        ])

        # ── Decoder ─────────────────────────────────────────────────────────
        dec_dims = list(reversed(dims))        # [768, 384, 192, 96]
        dec_res = list(reversed(enc_res))
        dec_dep = list(reversed(depths))
        dec_hds = list(reversed(num_heads))

        # Upsample: dec_dims[i] → dec_dims[i]//2  (= dec_dims[i+1])
        self.patch_expanding = nn.ModuleList([
            _PatchExpanding(dec_dims[i], dec_dims[i] // 2) for i in range(3)
        ])
        # After concat with same-depth encoder skip the channel count doubles;
        # project back to dec_dims[i+1].
        self.concat_proj = nn.ModuleList([
            nn.Linear(dec_dims[i], dec_dims[i + 1], bias=False) for i in range(3)
        ])
        self.dec_stages = nn.ModuleList([
            _SwinStage(dec_dims[i + 1], dec_res[i + 1], dec_dep[i + 1], dec_hds[i + 1], window_size)
            for i in range(3)
        ])

        # ── Final upsample × patch_size → full resolution ───────────────────
        # embed_dim tokens → (patch_size)² × (embed_dim//4) via linear +
        # pixel-shuffle
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_expand = nn.Linear(
            embed_dim, patch_size ** 2 * (embed_dim // 4), bias=False)
        self.head = nn.Conv2d(embed_dim // 4, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = x.shape

        # Patch embed → token sequence
        x, Hp, Wp = self.patch_embed(x)

        # Encoder (collect skip connections before each patch-merge)
        skips: List[Tuple[torch.Tensor, int, int]] = []
        Hc, Wc = Hp, Wp
        for i in range(4):
            x = self.enc_stages[i](x)
            if i < 3:
                skips.append((x, Hc, Wc))
                x, Hc, Wc = self.patch_merging[i](x, Hc, Wc)

        # Decoder (skip[2-i] reverses depth order)
        for i in range(3):
            x, Hc, Wc = self.patch_expanding[i](x, Hc, Wc)
            sx, sH, sW = skips[2 - i]
            x = self.concat_proj[i](torch.cat([x, sx], dim=-1))
            x = self.dec_stages[i](x)

        # Final upsample ×patch_size via pixel-shuffle
        x = self.final_expand(self.final_norm(x))    # (B, Hp*Wp, ps²*(C//4))
        ps = self.patch_size
        och = self.embed_dim // 4
        x = (
            x.view(B, Hp, Wp, ps, ps, och)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, och, H, W)
        )
        return self.head(x)


# ===========================================================================
# 5. LinkNet
# ===========================================================================


class _ResBlock(nn.Module):
    """Basic residual block with optional stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) + self.skip(x))


class _LinkDecBlock(nn.Module):
    """LinkNet decoder block: bottleneck → transposed-conv ×2 → project."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        mid = in_ch // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, mid, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LinkNet(nn.Module):
    """
    LinkNet for binary semantic segmentation.

    Lightweight encoder-decoder where skip connections are added element-wise
    (not concatenated), giving far fewer parameters than UNet while retaining
    spatial detail.  Encoder: 4 stages of residual blocks with stride-2
    downsampling.  Decoder: per-stage bottleneck → transposed-conv ×2 upsample
    → project, then element-wise add to the matching encoder skip (Chaurasia &
    Culurciello, "LinkNet", VCIP 2017).

    Args:
        in_channels   : Raster bands fed into the model (1 – 40).
        img_size      : Square spatial size of each input patch; must be one of
                        ``SUPPORTED_SIZES``.
        base_channels : Feature-map width at encoder stage 1 (default 64).
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 256,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        assert 1 <= in_channels <= MAX_IN_CHANNELS, (
            f"in_channels must be 1 – {MAX_IN_CHANNELS}, got {in_channels}."
        )
        assert img_size in SUPPORTED_SIZES, UNSUPPORTED_SIZE_WARNING

        c = base_channels
        # Initial projection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        # Encoder
        self.enc1 = nn.Sequential(_ResBlock(c, c), _ResBlock(c, c))
        self.enc2 = nn.Sequential(
            _ResBlock(
                c,
                c * 2,
                stride=2),
            _ResBlock(
                c * 2,
                c * 2))
        self.enc3 = nn.Sequential(
            _ResBlock(
                c * 2,
                c * 4,
                stride=2),
            _ResBlock(
                c * 4,
                c * 4))
        self.enc4 = nn.Sequential(
            _ResBlock(
                c * 4,
                c * 8,
                stride=2),
            _ResBlock(
                c * 8,
                c * 8))
        # Decoder (element-wise skip adds — no channel concatenation)
        self.dec4 = _LinkDecBlock(c * 8, c * 4)
        self.dec3 = _LinkDecBlock(c * 4, c * 2)
        self.dec2 = _LinkDecBlock(c * 2, c)
        self.dec1 = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        # Segmentation head (raw logit)
        self.head = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d = self.dec4(s4) + s3   # element-wise add
        d = self.dec3(d) + s2
        d = self.dec2(d) + s1
        d = self.dec1(d)
        return self.head(d)


# ===========================================================================
# 6. DeepLabV3+
# ===========================================================================

# ASPP dilation rates are scaled down for small tiles so the receptive field
# stays within the feature-map boundaries.
_ASPP_RATES: dict = {
    64: (2, 3, 4),
    128: (3, 6, 9),
    256: (6, 12, 18),
    512: (6, 12, 18),
    1024: (6, 12, 18),
}


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with global average-pooling branch."""

    def __init__(self, in_ch: int, out_ch: int,
                 rates: Tuple[int, ...]) -> None:
        super().__init__()
        # 1×1 conv + one dilated 3×3 per rate
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ),
            *[
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for r in rates
            ],
        ])
        # Global average-pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        n_branches = 1 + len(rates) + 1   # 1×1 + dilated + gap
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch * n_branches, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]
        parts = [b(x) for b in self.branches]
        gap = F.interpolate(
            self.gap(x),
            size=(
                H,
                W),
            mode="bilinear",
            align_corners=True)
        parts.append(gap)
        return self.proj(torch.cat(parts, dim=1))


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ for binary semantic segmentation.

    From-scratch encoder-decoder with Atrous Spatial Pyramid Pooling (ASPP)
    and a lightweight decoder that fuses high-level ASPP features with
    low-level encoder features (Chen et al., "Encoder-Decoder with Atrous
    Separable Convolution", ECCV 2018).  No pretrained backbone — the encoder
    is a plain 3-stage strided CNN, so any number of input bands is supported.

    ASPP dilation rates are selected automatically per ``img_size`` so that
    the receptive field stays meaningful even on 64 × 64 tiles.

    Args:
        in_channels   : Raster bands fed into the model (1 – 40).
        img_size      : Square spatial size of each input patch; must be one of
                        ``SUPPORTED_SIZES``.
        base_channels : Feature-map width at encoder stage 1 (default 64).
        aspp_out_ch   : Output channels of the ASPP module (default 256).
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 256,
        base_channels: int = 64,
        aspp_out_ch: int = 256,
    ) -> None:
        super().__init__()
        assert 1 <= in_channels <= MAX_IN_CHANNELS, (
            f"in_channels must be 1 – {MAX_IN_CHANNELS}, got {in_channels}."
        )
        assert img_size in SUPPORTED_SIZES, UNSUPPORTED_SIZE_WARNING

        c = base_channels

        # Encoder — output stride 8
        self.layer0 = nn.Sequential(                                          # → /1
            nn.Conv2d(in_channels, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(                                          # → /2  low-level feats
            nn.Conv2d(c, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(                                          # → /4
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(                                          # → /8  ASPP input
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
        )

        self.aspp = _ASPP(
            c * 4,
            out_ch=aspp_out_ch,
            rates=_ASPP_RATES[img_size])

        # Project low-level features to 48 ch (following the paper)
        self.low_proj = nn.Sequential(
            nn.Conv2d(c, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            DoubleConv(aspp_out_ch + 48, c * 2),
            DoubleConv(c * 2, c * 2),
        )
        self.head = nn.Conv2d(c * 2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low = self.layer1(self.layer0(x))      # /2 — low-level features
        h = self.layer3(self.layer2(low))    # /8 — high-level features

        aspp_out = self.aspp(h)
        aspp_up = F.interpolate(aspp_out,
                                size=low.shape[2:],
                                mode="bilinear",
                                align_corners=True)

        fused = torch.cat([aspp_up, self.low_proj(low)], dim=1)
        out = self.decoder(fused)
        out = F.interpolate(out,
                            size=x.shape[2:],
                            mode="bilinear",
                            align_corners=True)
        return self.head(out)


# ===========================================================================
# 7. SegFormer
# ===========================================================================


class _OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding via strided convolution (preserves local context)."""

    def __init__(
            self,
            in_ch: int,
            embed_dim: int,
            patch_size: int,
            stride: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim, patch_size,
            stride=stride, padding=patch_size // 2, bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        H, W = x.shape[2:]
        x = self.norm(x.flatten(2).transpose(1, 2))   # (B, H*W, C)
        return x, H, W


class _EfficientSelfAttn(nn.Module):
    """Multi-head self-attention with spatial-reduction ratio R for K and V."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, sr_ratio, stride=sr_ratio, bias=False)
            self.sr_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        Nh = self.num_heads
        q = self.q(x).reshape(B, N, Nh, C // Nh).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = self.sr(
                x.transpose(
                    1,
                    2).reshape(
                    B,
                    C,
                    H,
                    W)).flatten(2).transpose(
                1,
                2)
            x_ = self.sr_norm(x_)
        else:
            x_ = x
        kv = self.kv(x_).reshape(B, -1, 2, Nh, C // Nh).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _MixFFN(nn.Module):
    """Mix FFN: linear → depth-wise conv (local context) → GELU → linear."""

    def __init__(self, dim: int, expand: int = 4) -> None:
        super().__init__()
        hidden = dim * expand
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            3,
            padding=1,
            groups=hidden,
            bias=False)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x)
        B, N, C = x.shape
        x = F.gelu(
            self.dw(
                x.transpose(
                    1,
                    2).reshape(
                    B,
                    C,
                    H,
                    W)).flatten(2).transpose(
                1,
                2))
        return self.fc2(x)


class _MiTBlock(nn.Module):
    """One Mix Transformer block: layer-norm → efficient attention → Mix FFN."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _EfficientSelfAttn(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = _MixFFN(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class _MiTStage(nn.Module):
    """One hierarchical stage of the Mix Transformer encoder."""

    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        num_heads: int,
        depth: int,
        patch_size: int,
        stride: int,
        sr_ratio: int,
    ) -> None:
        super().__init__()
        self.patch_embed = _OverlapPatchEmbed(
            in_ch, embed_dim, patch_size, stride)
        self.blocks = nn.ModuleList([
            _MiTBlock(embed_dim, num_heads, sr_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, H, W = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        B, _, C = x.shape
        return x.transpose(1, 2).reshape(B, C, H, W)


class SegFormer(nn.Module):
    """
    SegFormer for binary semantic segmentation.

    Hierarchical Mix Transformer (MiT) encoder with four stages of increasing
    embedding dimension and efficient self-attention with spatial reduction.
    Decoder: all four feature maps are projected to a unified dimension,
    bilinearly upsampled to stage-0 resolution, concatenated, fused, then
    upsampled to full resolution (Xie et al., "SegFormer", NeurIPS 2021).

    Default hyper-parameters follow the lightweight MiT-B1 variant and work
    with all supported patch sizes including 64 × 64.  No pretrained weights
    are required — the model is fully compatible with arbitrary band counts.

    Args:
        in_channels  : Raster bands fed into the model (1 – 40).
        img_size     : Square spatial size of each input patch; must be one of
                       ``SUPPORTED_SIZES``.
        embed_dims   : Embedding dims per stage (default [32, 64, 160, 256]).
        num_heads    : Attention heads per stage (default [1, 2, 5, 8]).
        depths       : Transformer blocks per stage (default [2, 2, 2, 2]).
        sr_ratios    : Spatial-reduction ratios per stage (default [8, 4, 2, 1]).
        decoder_dim  : Unified decoder projection dimension (default 256).
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 256,
        embed_dims: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        depths: Optional[List[int]] = None,
        sr_ratios: Optional[List[int]] = None,
        decoder_dim: int = 256,
    ) -> None:
        super().__init__()
        assert 1 <= in_channels <= MAX_IN_CHANNELS, (
            f"in_channels must be 1 – {MAX_IN_CHANNELS}, got {in_channels}."
        )
        assert img_size in SUPPORTED_SIZES, UNSUPPORTED_SIZE_WARNING

        embed_dims = embed_dims or [32, 64, 160, 256]
        num_heads = num_heads or [1, 2, 5, 8]
        depths = depths or [2, 2, 2, 2]
        sr_ratios = sr_ratios or [8, 4, 2, 1]

        # MiT hierarchical encoder
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        in_chs = [in_channels] + embed_dims[:3]
        self.stages = nn.ModuleList([
            _MiTStage(
                in_chs[i], embed_dims[i], num_heads[i], depths[i],
                patch_sizes[i], strides[i], sr_ratios[i],
            )
            for i in range(4)
        ])

        # All-MLP decoder
        self.proj = nn.ModuleList([
            nn.Conv2d(embed_dims[i], decoder_dim, 1, bias=False) for i in range(4)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(decoder_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = x.shape

        feats: List[torch.Tensor] = []
        xi = x
        for stage in self.stages:
            xi = stage(xi)
            feats.append(xi)

        # Upsample all stages to stage-0 resolution (/4 of input), then fuse
        target = feats[0].shape[2:]
        outs = [
            F.interpolate(
                self.proj[i](
                    feats[i]),
                size=target,
                mode="bilinear",
                align_corners=False) for i in range(4)]
        out = self.fuse(torch.cat(outs, dim=1))
        out = F.interpolate(
            out,
            size=(
                H,
                W),
            mode="bilinear",
            align_corners=False)
        return self.head(out)


# ===========================================================================
# Factory
# ===========================================================================

_REGISTRY: dict = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "unet_pp": UNetPP,
    "swin_unet": SwinUNet,
    "linknet": LinkNet,
    "deeplabv3plus": DeepLabV3Plus,
    "segformer": SegFormer,
}

AVAILABLE_MODELS: List[str] = list(_REGISTRY.keys())
"""String keys accepted by ``build_model``."""


def build_model(
    name: str,
    in_channels: int,
    img_size: int,
    **kwargs,
) -> nn.Module:
    """Instantiate a segmentation model by name.

    Args:
        name        : Architecture key – one of ``AVAILABLE_MODELS``:
                      ``"unet"``, ``"attention_unet"``, ``"unet_pp"``, ``"swin_unet"``,
                      ``"linknet"``, ``"deeplabv3plus"``, ``"segformer"``.
        in_channels : Number of input raster bands (1 – 40).
        img_size    : Square patch size in pixels; must be in ``SUPPORTED_SIZES``.
        **kwargs    : Extra keyword arguments forwarded to the model constructor
                      (e.g. ``base_channels``, ``deep_supervision``, ``embed_dim``).

    Returns:
        An untrained ``nn.Module`` with a single-channel logit output.

    Raises:
        ValueError if *name* is not recognised.
        AssertionError if *in_channels* or *img_size* are out of range.

    Example::

        model = build_model("unet", in_channels=4, img_size=512)
        logits = model(torch.randn(2, 4, 512, 512))  # (2, 1, 512, 512)
        probs  = torch.sigmoid(logits)
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {AVAILABLE_MODELS}"
        )
    return _REGISTRY[key](in_channels=in_channels, img_size=img_size, **kwargs)
