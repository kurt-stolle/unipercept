"""
An implementation of the DICE loss function for segmentation based on a padded tensor for separate background 
and foreground classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import typing_extensions as TX

from .mixins import ScaledLossMixin, StableLossMixin

__all__ = ["WeightedStuffDiceLoss", "WeightedThingDiceLoss", "WeightedThingFocalLoss"]


class WeightedStuffDiceLoss(StableLossMixin, ScaledLossMixin, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @TX.override
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_num: int,
        y_mask: torch.Optional[torch.Tensor],
        index_mask: torch.Tensor,
    ) -> torch.Tensor:
        if y_num == 0:
            return x.mean() * 0.0

        _, _, h, w = y.shape

        x = x.reshape(-1, h, w)[index_mask, ...]
        y = y.reshape(-1, h, w)[index_mask, ...]

        x = torch.sigmoid(x)

        x = x.reshape(int(y_num), h * w)
        y = y.reshape(int(y_num), h * w)

        if y_mask is not None:
            y_mask = y_mask.reshape(-1, h, w)[index_mask, ...]
            y_mask = y_mask.reshape(int(y_num), h * w)

        return self._dice(x, y, y_mask, None)

    @torch.jit.export
    def _dice(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_valid: torch.Optional[torch.Tensor] = None,
        weights: torch.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if y_valid is not None:
            invalid = torch.zeros_like(x)
            x = torch.where(y_valid, x, invalid)
            y = torch.where(y_valid, y, invalid)

        loss_part = (x**2).sum(dim=-1) + (y**2).sum(dim=-1)
        loss = 1.0 - 2.0 * (y * x).sum(dim=-1) / loss_part.clamp(self.eps)

        if weights is not None:
            loss = loss * weights

        return loss.sum() * self.scale


class WeightedThingDiceLoss(WeightedStuffDiceLoss):
    @TX.override
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_num: int,
        y_mask: torch.Optional[torch.Tensor],
        index_mask: torch.Tensor,
        instance_num: int,
        weight_num: int,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if y_num == 0:
            return x.sigmoid().mean() * 0.0

        n, _, h, w = y.shape

        weights = weights.float()

        x = x.reshape(n, instance_num, weight_num, h, w)
        x = x.reshape(-1, weight_num, h, w)[index_mask, ...]

        y = y.unsqueeze(2).expand(n, instance_num, weight_num, h, w)
        y = y.reshape(-1, weight_num, h, w)[index_mask, ...]

        weights = weights.reshape(-1, weight_num)[index_mask, ...]
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(self.eps)

        x = torch.sigmoid(x)

        x = x.reshape(int(y_num), weight_num, h * w)
        y = y.reshape(int(y_num), weight_num, h * w)

        if y_mask is not None:
            y_mask = y_mask.unsqueeze(2).expand(n, instance_num, weight_num, h, w)
            y_mask = y_mask.reshape(-1, weight_num, h, w)[index_mask, ...]
            y_mask = y_mask.reshape(int(y_num), weight_num, h * w)

        return self._dice(x, y, y_mask, weights)


class WeightedThingFocalLoss(ScaledLossMixin, StableLossMixin, nn.Module):
    alpha: torch.jit.Final[float]
    gamma: torch.jit.Final[float]

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma

    @TX.override
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        y_num: int,
        index_mask: torch.Tensor,
        instance_num: int,
        weights: torch.Tensor,
        weight_num: int,
    ) -> torch.Tensor:
        """
        Weighted version of Dice Loss used in PanopticFCN for multi-positive optimization.

        Adapted from: https://github.com/DdeGeus/PanopticFCN-IBS/blob/master/panopticfcn/loss.py

        Parameters
        ----------
        x
            prediction logits
        y
            segmentation target for Things or Stuff,
        gt_num
            ground truth number for Things or Stuff,
        index_mask
            positive index mask for Things or Stuff,
        instance_num
            instance number of Things or Stuff,
        weighted_val
            values of k positives,
        weighted_num
            number k for weighted loss,
        """
        # avoid Nan
        if y_num == 0:
            loss = x.sigmoid().mean() + self.eps
            return loss * y_num

        n, _, h, w = y.shape
        x = x.reshape(n, instance_num, weight_num, h, w)
        x = x.reshape(-1, weight_num, h, w)[index_mask, ...]
        y = y.unsqueeze(2).expand(n, instance_num, weight_num, h, w)
        y = y.reshape(-1, weight_num, h, w)[index_mask, ...]
        weights = weights.reshape(-1, weight_num)[index_mask, ...]
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=self.eps)
        x = x.reshape(int(y_num), weight_num, h * w)
        y = y.reshape(int(y_num), weight_num, h * w)

        p = torch.sigmoid(x)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(x, y, reduction="none")

        p_t = p * y + (1 - p) * (1 - y)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
            loss = alpha_t * loss

        loss = loss.mean(dim=-1)
        loss = loss * weights

        return loss.sum() * self.scale
