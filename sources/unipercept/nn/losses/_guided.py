"""
Implements segmentation-to-depth and vice-versa losses.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch import nn
from typing_extensions import override

import unipercept.data.tensors as _P
from unipercept.utils.logutils import get_logger

__all__ = ["DGPLoss", "PGTLoss", "PGSLoss"]

logger = get_logger(__name__)


class DGPLoss(nn.Module):
    """
    Implements a depth-guided panoptic loss (DGP) loss
    """

    def __init__(self, tau=10, patch_size=5, eps=1e-8):
        super().__init__()

        self.tau = tau
        self.patch_size = patch_size
        self.eps = eps

    def get_patches(self, tensor: torch.Tensor):
        # Using unfold to get patches
        k = self.patch_size
        patches = tensor.unfold(2, k, 1).unfold(3, k, 1)
        # Reshape patches to separate the patch dimension from spatial dimensions
        patches = patches.permute(0, 1, 4, 5, 2, 3).reshape(tensor.size(0), tensor.size(1), k, k, -1)
        return patches

    @override
    def forward(self, semantic_features: torch.Tensor, depth_gt: torch.Tensor):
        assert depth_gt.ndim == 4, f"{depth_gt.shape=}"
        assert semantic_features.ndim == 4, f"{semantic_features.shape=}"
        assert semantic_features.shape[-2:] == depth_gt.shape[-2:], f"{semantic_features.shape=}, {depth_gt.shape=}"

        # Convert target labels to floating point and apply nearest neighbors downsampling

        # Get patches
        sem_patches = self.get_patches(semantic_features)
        depth_patches = self.get_patches(1 / (depth_gt + 1e-6))

        # Extract center from each patch
        center_idx = self.patch_size // 2
        sem_center = sem_patches[:, :, center_idx, center_idx, :]
        depth_center = depth_patches[:, 0, center_idx, center_idx, :]

        # Compute loss for all patches and centers
        depth_diff = torch.abs(depth_center[:, None, None, :] - depth_patches[:, 0])
        sem_diff = torch.norm(sem_center[:, :, None, None, :] - sem_patches, dim=1)

        # Apply the loss formula
        depth_diff = torch.clamp(depth_diff, min=self.eps)
        sem_diff = torch.clamp(sem_diff, min=self.eps)
        loss = (torch.expm1(-depth_diff / self.tau) + 1) * (torch.expm1(-(sem_diff**2)) + 1)
        mask = (depth_diff > self.eps) & (sem_diff > self.eps) & (depth_patches > 0.0)
        loss = loss * mask

        # Exclude center pixels from the loss computation (equivalent to dx == 0 and dy == 0 in the original code)
        loss[:, :, center_idx, center_idx] = 0

        # Average the loss over patch dimensions, then over spatial dimensions
        loss = torch.mean(loss, dim=(2, 3))
        total_loss = torch.mean(loss)

        return total_loss


class PGTLoss(nn.Module):
    """
    Panoptic-guided Triplet Loss (PGT) loss

    Paper: https://arxiv.org/abs/2210.07577
    """

    def __init__(self, p_height=5, p_width=5, margin=0.3):
        super().__init__()

        self.p_height = p_height
        self.p_width = p_width
        self.margin = margin
        self.threshold = p_height - 1

    def build_patches(self, input: torch.Tensor, p_height: int, p_width: int) -> torch.Tensor:
        """
        Splits tensor into N, p_height x p_width blocks.
        """

        patches = input.unfold(2, p_height, p_height)
        patches = patches.unfold(3, p_width, p_width)
        patches = patches.contiguous()
        patches = patches.view(input.shape[0], input.shape[1], -1, p_height, p_width)

        return patches

    @override
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # Convert target labels to floating point and apply nearest neighbors downsampling
        target = target.float()
        target = nn.functional.interpolate(target.unsqueeze(1), size=output.shape[-2:], mode="nearest")

        # Split both depth estimated output and panoptic label into NxN patches
        # P ~= (H * W) / (5 * 5)
        output_patches = self.build_patches(output, self.p_height, self.p_width)  # N x C x P x 5 x 5
        target_patches = self.build_patches(target, self.p_height, self.p_width)  # N x 1 x P x 5 x 5

        # Discard patches that do not intersect panoptic contours (set them to NaN)
        # Divide each batch to patches
        patch_len = target_patches.flatten(-2, -1).shape[-1]  # N x C x P x 25 (flatten tensor)
        patch_sum = torch.sum(target_patches.flatten(-2, -1), dim=-1)  # N x C x P
        # take the first value of each patch for sanity check
        is_equal = patch_len * target_patches.flatten(-2, -1)[..., 0]  # N x C x P
        valid_target_patches_idx = patch_sum != is_equal  # N x C x P
        valid_target_patches_idx_ext = (
            valid_target_patches_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.p_height, self.p_width)
        )  # N x 1 x P x 5 x 5
        target_patches.masked_fill_(valid_target_patches_idx_ext == False, torch.nan)
        valid_target_patches_idx_ext = valid_target_patches_idx_ext.repeat(
            1, output.shape[1], 1, 1, 1
        )  # N x C x P x 5 x 5
        output_patches.masked_fill_(valid_target_patches_idx_ext == False, torch.nan)

        # Calculate anchors of output and target
        target_anchor = (
            target_patches[..., target_patches.shape[-2] // 2, target_patches.shape[-1] // 2]
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # N x C x P x 1 x 1
        output_anchor = (
            output_patches[..., output_patches.shape[-2] // 2, output_patches.shape[-1] // 2]
            .unsqueeze(-1)
            .unsqueeze(-1)
        )  # N x C x P x 1 x 1

        # Calculate mask of positive and negative features
        mask_pos = ((target_patches == target_anchor) & ~torch.isnan(target_patches)).int()  # N x C x P x 5 x 5
        mask_neg = ((target_patches != target_anchor) & ~torch.isnan(target_patches)).int()  # N x C x P x 5 x 5

        # Calculate filtered output (depth) values given the mask
        output_pos = output_patches * mask_pos  # N x C x P x 5 x 5
        output_neg = output_patches * mask_neg  # N x C x P x 5 x 5

        # Estimate number of positive and negative features
        target_pos_num = mask_pos.count_nonzero(dim=(-2, -1)).float()  # N x C x P x 1 x 1
        target_neg_num = mask_neg.count_nonzero(dim=(-2, -1)).float()  # N x C x P x 1 x 1

        # Distance terms for all patches
        distance_pos = (1 / target_pos_num) * torch.norm(
            input=torch.nan_to_num(output_anchor * mask_pos - output_pos), p=2, dim=(-2, -1)
        )  # N x C x P x 1
        distance_neg = (1 / target_neg_num) * torch.norm(
            input=torch.nan_to_num(output_anchor * mask_neg - output_neg), p=2, dim=(-2, -1)
        )  # N x C x P x 1

        # Total loss for all patches
        patch_losses = torch.maximum(
            torch.zeros(size=(distance_pos.shape[0], distance_pos.shape[1], 1), device=target.device),
            distance_pos + self.margin - distance_neg,
        )  # N x C x P

        # Get mask of patches with positive features and negative features larger than threshold and get filtered patch losses
        pos_gt_threshold_mask = target_pos_num > self.threshold  # N x C x P
        neg_gt_threshold_mask = target_neg_num > self.threshold  # N x C x P
        num_gt_threshold_mask = pos_gt_threshold_mask & neg_gt_threshold_mask  # N x C x P

        patch_losses = torch.masked_select(patch_losses, num_gt_threshold_mask)  # N x C x P'

        # Calculate overall loss
        total_loss = torch.sum(patch_losses) / patch_losses.shape[0]
        return total_loss


class PGSLoss(nn.Module):
    """
    Panoptic-guided Smoothness Loss (PGS)

    Paper: https://arxiv.org/abs/2210.07577
    """

    def __init__(self):
        super().__init__()

    @override
    def forward(
        self, output: torch.Tensor, target: torch.Tensor
    ):  # NOTE:  target is panoptic mask, output is norm disparity
        # Compute the Iverson bracket for adjacent pixels along the x-dimension
        panoptic_diff_x = (torch.diff(target, dim=-1) != 0).int()

        # Compute the Iverson bracket for adjacent pixels along the y-dimension
        panoptic_diff_y = (torch.diff(target, dim=-2) != 0).int()

        # Compute the partial disp derivative along the x-axis
        disp_diff_x = torch.diff(output, dim=-1)

        # Compute the partial disp derivative along the y-axis
        disp_diff_y = torch.diff(output, dim=-2)

        loss = torch.mean(torch.abs(disp_diff_x) * (1 - panoptic_diff_x)) + torch.mean(
            torch.abs(disp_diff_y) * (1 - panoptic_diff_y)
        )

        return loss
