"""
An implementation of the DICE loss function for segmentation based on a padded tensor for separate background 
and foreground classes.
"""

# from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from typing_extensions import override

from .mixins import StableLossMixin, ScaledLossMixin


class WeightedStuffDiceLoss(StableLossMixin, ScaledLossMixin, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    @torch.jit.script_if_tracing
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
    @torch.jit.script_if_tracing
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
    @override
    @torch.jit.script_if_tracing
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
