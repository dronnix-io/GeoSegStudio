"""
module: DL/losses.py

Loss functions for binary semantic segmentation.

All losses expect:
  logits  : (B, 1, H, W)  raw model output (before sigmoid)
  targets : (B, 1, H, W)  binary ground truth, values in {0.0, 1.0}

Supported keys (selected via the Training Configuration section in Tab 2):
  "bce"      — Binary Cross-Entropy with logits
  "dice"     — Dice loss  (1 − Dice coefficient)
  "bce_dice" — 0.5 * BCE + 0.5 * Dice  (recommended default)

Public API
----------
build_loss(name: str) -> nn.Module
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """Binary Cross-Entropy with logits (numerically stable)."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets)


class DiceLoss(nn.Module):
    """
    Soft Dice loss: 1 − (2 * |P ∩ T| + ε) / (|P| + |T| + ε)

    Works on flattened predictions and targets so it is resolution-agnostic.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )


class BCEDiceLoss(nn.Module):
    """Equal-weight combination of BCE and Dice losses."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.bce  = BCELoss()
        self.dice = DiceLoss(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LOSSES = {
    "bce":      BCELoss,
    "dice":     DiceLoss,
    "bce_dice": BCEDiceLoss,
}


def build_loss(name: str) -> nn.Module:
    """
    Instantiates the loss function for the given key.

    Parameters
    ----------
    name : str
        One of "bce", "dice", "bce_dice".

    Returns
    -------
    nn.Module
    """
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss {name!r}. Choose from {list(_LOSSES)}")
    return _LOSSES[name]()
