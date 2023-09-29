from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override
from uniutils.mask import masks_to_boxes

from .loss_utils import NumStableLoss


def compute_sile(x: Tensor, y: Tensor, num: int) -> Tensor:
    """Scale invariant logarithmic error."""
    log_err = torch.log1p(x) - torch.log1p(y)

    sile_1 = log_err.square().sum() / num
    sile_2 = log_err.sum() / (num**2)

    return sile_1 - sile_2


def compute_rel(x: Tensor, y: Tensor, num: int) -> tuple[Tensor, Tensor]:
    """Square relative error and absolute relative error."""
    err = x - y
    err_rel = err / y.clamp(1e-6)
    are = err_rel.abs().sum() / num

    sre = err_rel.square().sum() / num
    sre = sre.clamp(1e-8).sqrt()

    return are, sre


class DepthLoss(nn.Module):
    def __init__(self, weight_sile=1.0, weight_are=1.0, weight_sre=1.0, **kwargs):
        super().__init__(**kwargs)

        self.weight_sile = weight_sile
        self.weight_are = weight_are
        self.weight_sre = weight_sre

    @override
    def forward(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        *,
        mask: torch.Tensor,
    ) -> Tensor:
        assert mask.dtype == torch.bool, mask.dtype
        assert mask.ndim >= 3, mask.ndim

        mask = mask & (y > 0)
        if not mask.any():
            return x.sum() * 0.0

        y = y[mask]
        x = x[mask]
        n = y.numel()

        sile = compute_sile(x, y, n)
        are, sre = compute_rel(x, y, n)

        loss = sile * self.weight_sile + are * self.weight_are + sre * self.weight_sre

        return loss


# Aliases for legacy code
DepthInstanceLoss = DepthLoss
DepthFlatLoss = DepthLoss


class DepthSmoothLoss(NumStableLoss):
    """
    Compute the depth smoothness loss, defined as the weighted smoothness
    of the inverse depth.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.padded = False

    @override
    def forward(self, images: Tensor, depths: Tensor, mask: Tensor | None) -> Tensor:
        if len(images) == 0:
            return depths.sum()

        if mask is None:
            mask = torch.ones_like(images).bool()

        # compute inverse depths
        # idepths = 1 / (depths / self.depth_max).clamp(self.eps)
        # idepths = depths.float() / self.depth_max
        # idepths = depths
        idepths = 1 / self._nsb(depths, is_small=True)

        # compute the gradients
        idepth_dx: Tensor = self._gradient_x(idepths)
        idepth_dy: Tensor = self._gradient_y(idepths)
        image_dx: Tensor = self._gradient_x(images)
        image_dy: Tensor = self._gradient_y(images)

        # compute image weights
        weights_x: Tensor = torch.exp(-torch.mean(torch.abs(image_dx) + self.eps, dim=1, keepdim=True))
        weights_y: Tensor = torch.exp(-torch.mean(torch.abs(image_dy) + self.eps, dim=1, keepdim=True))

        # apply image weights to depth
        smoothness_x: Tensor = torch.abs(idepth_dx * weights_x)
        smoothness_y: Tensor = torch.abs(idepth_dy * weights_y)

        # compute shifted masks
        if self.padded:
            h, w = mask.shape[-2], mask.shape[-1]

            mask_boxes = masks_to_boxes(mask.squeeze(1).long())

            assert all(mask_boxes[:, 2] < w)
            assert all(mask_boxes[:, 3] < h)

            mask_margin = 1 / 10
            mask_boxes[:, 0] = (mask_boxes[:, 0] - int(w * mask_margin)).clamp(min=0)
            mask_boxes[:, 1] = (mask_boxes[:, 1] - int(h * mask_margin)).clamp(min=0)
            mask_boxes[:, 2] = (mask_boxes[:, 2] + int(w * mask_margin)).clamp(max=w)
            mask_boxes[:, 3] = (mask_boxes[:, 3] + int(h * mask_margin)).clamp(max=h)

            mask_padded = torch.full_like(mask, False)
            for i, (x_min, y_min, x_max, y_max) in enumerate(mask_boxes):
                mask_padded[i, 0, y_min:y_max, x_min:x_max] = True

            mask_x = mask_padded[:, :, :, 1:]
            mask_y = mask_padded[:, :, 1:, :]
        else:
            mask_x = mask[:, :, :, 1:]
            mask_y = mask[:, :, 1:, :]

        # loss for x and y
        loss_x = smoothness_x[mask_x].mean()
        loss_y = smoothness_y[mask_y].mean()

        assert loss_x.isfinite()
        assert loss_y.isfinite()

        return loss_x + loss_y

    def _gradient_x(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def _gradient_y(self, img: Tensor) -> Tensor:
        if len(img.shape) != 4:
            raise AssertionError(img.shape)
        return img[:, :, :-1, :] - img[:, :, 1:, :]
