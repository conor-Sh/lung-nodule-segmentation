from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice = dice_loss(pred, target)
        return bce_loss + dice


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred = (torch.sigmoid(pred) > threshold).float()
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    score = (2.0 * intersection + smooth) / (denom + smooth)
    return score.mean()
