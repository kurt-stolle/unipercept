from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from typing_extensions import override

from .loss_utils import NumStableLoss, ReducableLoss


class WeightedStuffDiceLoss(NumStableLoss, ReducableLoss):
    def __init__(self, *, reduction="sum", **kwargs):
        super().__init__(**kwargs)

        self.reduction = reduction

        assert reduction != "none"

    @override
    def forward(
        self,
        x: Tensor,
        y: Tensor,
        y_num: int,
        y_mask: Tensor | None,
        index_mask: Tensor,
    ) -> Tensor:
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

        return self._dice(x=x, y=y, y_valid=y_mask, weights=None)

    @torch.jit.export
    def _dice(
        self,
        x: Tensor,
        y: Tensor,
        y_valid: Tensor | None,
        weights: Tensor | None,
    ) -> Tensor:
        if y_valid is not None:
            invalid = torch.zeros_like(x)
            x = torch.where(y_valid, x, invalid)
            y = torch.where(y_valid, y, invalid)

        loss_part = (x**2).sum(dim=-1) + (y**2).sum(dim=-1)
        loss = 1.0 - 2.0 * (y * x).sum(dim=-1) / self._nsb(loss_part)

        if weights is not None:
            loss = loss * weights

        return self._reduce(loss)


class WeightedThingDiceLoss(WeightedStuffDiceLoss):
    @override
    def forward(
        self,
        x: Tensor,
        y: Tensor,
        y_num: int,
        y_mask: Tensor | None,
        index_mask: Tensor,
        instance_num: int,
        weight_num: int,
        weights: Tensor,
    ) -> Tensor:
        if y_num == 0:
            return x.sigmoid().mean() * 0.0

        n, _, h, w = y.shape

        weights = weights.float()

        x = x.reshape(n, instance_num, weight_num, h, w)
        x = x.reshape(-1, weight_num, h, w)[index_mask, ...]

        y = y.unsqueeze(2).expand(n, instance_num, weight_num, h, w)
        y = y.reshape(-1, weight_num, h, w)[index_mask, ...]

        weights = weights.reshape(-1, weight_num)[index_mask, ...]
        weights = weights / self._nsb(weights.sum(dim=-1, keepdim=True))

        x = torch.sigmoid(x)

        x = x.reshape(int(y_num), weight_num, h * w)
        y = y.reshape(int(y_num), weight_num, h * w)

        if y_mask is not None:
            y_mask = y_mask.unsqueeze(2).expand(n, instance_num, weight_num, h, w)
            y_mask = y_mask.reshape(-1, weight_num, h, w)[index_mask, ...]
            y_mask = y_mask.reshape(int(y_num), weight_num, h * w)

        return self._dice(x=x, y=y, y_valid=y_mask, weights=weights)
