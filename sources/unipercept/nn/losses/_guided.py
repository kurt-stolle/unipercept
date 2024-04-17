"""
Implements segmentation-to-depth and vice-versa losses.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from einops import reduce
from torch import nn
from typing_extensions import override

from unipercept.nn.losses.functional import split_into_patches
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
        loss, mask = depth_guided_segmentation_loss(
            seg_feat, dep_true, self.eps, self.tau, self.patch_size
        )
        loss = torch.masked_select(loss, mask).mean()

        return loss * self.scale


def depth_guided_segmentation_loss(
    seg_feat: torch.Tensor,
    dep_true: torch.Tensor,
    eps: float,
    tau: int,
    patch_size: int,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    seg_patch = split_into_patches(seg_feat, (patch_size, patch_size))
    dep_patch = split_into_patches(dep_true, (patch_size, patch_size))
    dep_valid = dep_patch > eps

    center_idx = patch_size // 2

    # Extract center from each patch
    with torch.no_grad():
        seg_center = seg_patch[:, :, :, center_idx, center_idx].contiguous()
        seg_center.unsqueeze_(-1).unsqueeze_(-1)
        depth_center = dep_patch[:, :, :, center_idx, center_idx].contiguous()
        depth_center.unsqueeze_(-1).unsqueeze_(-1)

    # Compute loss for all patches and centers
    # TODO: Stability of the loss function
    dep_diff = torch.abs(depth_center - dep_patch)  # .clamp_(min=eps)
    seg_diff = torch.norm(seg_center - seg_patch, dim=1)  # .clamp(min=eps)
    loss = torch.exp(-dep_diff / tau) * torch.exp(-(seg_diff**2))

    # Compute the mask for which the loss function is valid
    with torch.no_grad():
        mask = (dep_diff > eps) & (seg_diff > eps) & dep_valid
        mask[:, :, :, center_idx, center_idx] = False

    return loss, mask


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
        self.threshold = max(1, min(self.patch_width, self.patch_height) // 2 - 1)

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


def segmentation_guided_triplet_loss(
    dep_feat: torch.Tensor,
    seg_true: torch.Tensor,
    margin: float,
    threshold: int,
    patch_height: int,
    patch_width: int,
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    if seg_true.ndim != dep_feat.ndim:
        seg_true = seg_true.unsqueeze(1)

    assert (
        dep_feat.ndim == 4
    ), f"Expected features as B x C x H x W, got: {dep_feat.shape} for segmentation {dep_feat.shape}"
    assert (
        seg_true.ndim == dep_feat.ndim
    ), f"Expected segmentatino as B x 1 x H x W, got: {seg_true.shape} for depth features {dep_feat.shape}!"

    with torch.no_grad():
        seg_true = nn.functional.interpolate(
            seg_true.float(),
            size=dep_feat.shape[-2:],
            mode="nearest-exact",
        )

    # Split both depth estimated output and panoptic label into NxN patches
    # P ~= (H * W) / (5 * 5)
    seg_patch = split_into_patches(
        seg_true, (patch_height, patch_width)  # , (patch_height, patch_width)
    )  # B x 1 x P x 5 x 5
    dep_patch = split_into_patches(
        dep_feat, (patch_height, patch_width)  # , (patch_height, patch_width)
    )  # B x C x P x 5 x 5

    # Discard patches that have a panoptic value below 0 (ignore) or that all have the same class (no panoptic contours)
    patch_min = reduce(seg_patch, "n c p h w -> n c p () ()", "min")
    patch_max = reduce(seg_patch, "n c p h w -> n c p () ()", "max")
    # patch_min = seg_patch.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    # patch_max = seg_patch.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    patch_valid = (patch_min >= 0) & (patch_min != patch_max)  # N x C x P x 1 x 1

    patch_center_i = patch_height // 2
    patch_center_j = patch_width // 2
    # Calculate anchors of output and target
    # N x C x P x 1 x 1
    target_anchor = (
        seg_patch[..., patch_center_i, patch_center_j].unsqueeze(-1).unsqueeze(-1)
    )
    output_anchor = (
        dep_patch[..., patch_center_i, patch_center_j].unsqueeze(-1).unsqueeze(-1)
    )

    # Calculate mask of positive and negative features
    # N x C x P x 5 x 5
    mask_pos = ((seg_patch == target_anchor) & patch_valid).int()
    mask_neg = ((seg_patch != target_anchor) & patch_valid).int()

    # Calculate filtered output (depth) values given the mask
    # N x C x P x 5 x 5
    output_pos = dep_patch * mask_pos
    output_neg = dep_patch * mask_neg

    # Count the amount of positive and negative features
    # N x C x P x 1 x 1
    target_pos_num = mask_pos.count_nonzero(dim=(-2, -1)).float()
    target_neg_num = mask_neg.count_nonzero(dim=(-2, -1)).float()

    # Compute the distance (l2) between the anchor and the positive/negative features
    distance_pos = torch.norm(
        output_anchor * mask_pos - output_pos, p=2, dim=(-2, -1)
    ) / target_pos_num.clamp(1)
    distance_neg = torch.norm(
        output_anchor * mask_neg - output_neg, p=2, dim=(-2, -1)
    ) / target_neg_num.clamp(1)

    # Total loss for all patches
    patch_losses = distance_pos + margin - distance_neg
    patch_losses = torch.where(patch_losses >= 0, patch_losses, 0)

    # Get mask of patches with positive features and negative features larger than threshold and get filtered patch losses
    # N x C x P
    mask_pos = target_pos_num > threshold
    mask_neg = target_neg_num > threshold
    mask = mask_pos & mask_neg

    return patch_losses, mask


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
