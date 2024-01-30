"""
Implements segmentation-to-depth and vice-versa losses.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from torch import nn
from typing_extensions import override

from unipercept.nn.losses.functional import (
    depth_guided_segmentation_loss,
    segmentation_guided_triplet_loss,
)
from unipercept.nn.losses.mixins import ScaledLossMixin, StableLossMixin

__all__ = ["DGPLoss", "PGTLoss", "PGSLoss"]


class DGPLoss(StableLossMixin, ScaledLossMixin, nn.Module):
    """
    Implements a depth-guided panoptic loss (DGP) loss
    """

    tau: T.Final[int]
    patch_size: T.Final[int]

    def __init__(self, *, tau=10, patch_size=5, **kwargs):
        super().__init__(**kwargs)

        self.tau = tau
        self.patch_size = patch_size

    @override
    def forward(self, seg_feat: torch.Tensor, dep_true: torch.Tensor):
        # assert dep_true.ndim == 4, f"{dep_true.shape}"
        # assert dep_true.shape[1] == 1, f"{dep_true.shape}"
        # assert seg_feat.ndim == 4, f"{seg_feat.shape}"
        # assert seg_feat.shape[-2:] == dep_true.shape[-2:], f"{seg_feat.shape}, {dep_true.shape}"

        # Get patches
        loss, mask = depth_guided_segmentation_loss(
            seg_feat, dep_true, self.eps, self.tau, self.patch_size
        )

        # if not mask.any():
        #     return seg_feat.mean() * 0.0

        # Average the loss over patch dimensions, then over spatial dimensions
        loss = torch.masked_select(loss, mask).mean()

        return loss * self.scale


class PGTLoss(StableLossMixin, ScaledLossMixin, nn.Module):
    """
    Panoptic-guided Triplet Loss (PGT) loss

    Paper: https://arxiv.org/abs/2210.07577
    """

    patch_width: T.Final[int]
    patch_height: T.Final[int]
    margin: T.Final[float]
    threshold: T.Final[int]

    def __init__(
        self, *, patch_size: T.Tuple[int, int] = (5, 5), margin=0.33, **kwargs
    ):
        super().__init__(**kwargs)

        self.patch_width, self.patch_height = patch_size
        self.margin = margin
        self.threshold = min(self.patch_width, self.patch_height) - 1

    @override
    def forward(self, dep_feat: torch.Tensor, seg_true: torch.Tensor):
        loss, mask = segmentation_guided_triplet_loss(
            dep_feat,
            seg_true,
            self.margin,
            self.threshold,
            self.patch_height,
            self.patch_width,
        )
        loss = torch.masked_select(loss, mask)  # N x C x P'

        # Calculate overall loss
        return loss.mean() * self.scale


class PGSLoss(ScaledLossMixin, nn.Module):
    """
    Panoptic-guided Smoothness Loss (PGS)

    Paper: https://arxiv.org/abs/2210.07577
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def forward(self, disparity: torch.Tensor, panoptic: torch.Tensor):
        # Compute the Iverson bracket for adjacent pixels along the x-dimension
        panoptic_diff_x = (torch.diff(panoptic, dim=-1) != 0).int()

        # Compute the Iverson bracket for adjacent pixels along the y-dimension
        panoptic_diff_y = (torch.diff(panoptic, dim=-2) != 0).int()

        # Compute the partial disp derivative along the x-axis
        disp_diff_x = torch.diff(disparity, dim=-1)

        # Compute the partial disp derivative along the y-axis
        disp_diff_y = torch.diff(disparity, dim=-2)

        loss_x = torch.mean(torch.abs(disp_diff_x) * (1 - panoptic_diff_x))
        loss_y = torch.mean(torch.abs(disp_diff_y) * (1 - panoptic_diff_y))

        return self.scale * (loss_x + loss_y)
