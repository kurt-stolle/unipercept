"""
Implements common functional operations for losses.
"""

from __future__ import annotations

import math
import typing as T

import torch
import torch.nn as nn

#####################
# General utilities #
#####################


def split_into_patches(
    x: torch.Tensor,
    sizes: T.Tuple[int, int],
    strides: T.Optional[T.Tuple[int, int]] = None,
) -> torch.Tensor:
    r"""
    Splits tensor into N, p_height x p_width blocks.
    """

    if strides is None:
        strides = (sizes[0], sizes[1])

    batch_size, channels, _, _ = x.shape

    for dim, (size, stride) in enumerate(zip(sizes, strides)):
        x = x.unfold(dim + 2, size, stride)

    x = x.reshape(batch_size, channels, -1, sizes[0], sizes[1])
    return x  # B x C x P x H x W


##############################
# Depth related loss metrics #
##############################


def scale_invariant_logarithmic_error(
    x: torch.Tensor, y: torch.Tensor, num: int, eps: float
) -> torch.Tensor:
    r"""
    Scale invariant logarithmic error.
    """
    log_err = torch.log(x + eps) - torch.log(y + eps)
    # log_err = torch.log1p(x) - torch.log1p(y)

    num_2 = num**2

    # sile_1 = log_err.square().sum()/num
    # sile_2 = log_err.sum().square()  / num_2

    sile_1 = (math.sqrt(num) * log_err).square().sum()
    sile_2 = log_err.sum().square()

    return (sile_1 - sile_2.clamp(max=sile_1)) / num_2


def relative_absolute_squared_error(
    x: torch.Tensor, y: torch.Tensor, num: int, eps: float
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Square relative error and absolute relative error.
    """
    err = x - y
    err_rel = err / y.clamp(eps)
    are = err_rel.abs().sum() / num

    sre = err_rel.square().sum() / num
    sre = sre.clamp(eps).sqrt()

    return are, sre


###################################
# Depth and segmentation guidance #
###################################


def depth_guided_segmentation_loss(
    seg_feat: torch.Tensor,
    dep_true: torch.Tensor,
    eps: float,
    tau: int,
    patch_size: int,
):
    seg_patch = split_into_patches(seg_feat, (patch_size, patch_size))
    dep_patch = split_into_patches(dep_true, (patch_size, patch_size))
    dep_valid = dep_patch > eps

    center_idx = patch_size // 2

    # Extract center from each patch
    with torch.no_grad():
        sem_center = seg_patch[:, :, :, center_idx, center_idx].contiguous()
        sem_center.unsqueeze_(-1).unsqueeze_(-1)
        depth_center = dep_patch[:, :, :, center_idx, center_idx].contiguous()
        depth_center.unsqueeze_(-1).unsqueeze_(-1)

    # Compute loss for all patches and centers
    dep_diff = torch.abs(depth_center - dep_patch).clamp_(min=eps)
    sem_diff = torch.norm(sem_center - seg_patch, dim=1).clamp(min=eps)
    loss = torch.exp(-dep_diff / tau) * torch.exp(-(sem_diff**2))

    # Compute the mask for which the loss function is valid
    with torch.no_grad():
        mask = (dep_diff > eps) & (sem_diff > eps) & dep_valid
        mask[:, :, :, center_idx, center_idx] = False

    return loss, mask


def segmentation_guided_triplet_loss(
    dep_feat: torch.Tensor,
    seg_true: torch.Tensor,
    margin: float,
    threshold: int,
    patch_height: int,
    patch_width: int,
):
    # if seg_true.ndim != dep_feat.ndim:
    #     seg_true.unsqueeze_(1)

    with torch.no_grad():
        # seg_true = seg_true.float()
        seg_true = nn.functional.interpolate(
            seg_true[:, None, ...].float(),
            size=dep_feat.shape[-2:],
            mode="nearest-exact",
        )

    # Split both depth estimated output and panoptic label into NxN patches
    # P ~= (H * W) / (5 * 5)
    seg_patch = split_into_patches(
        seg_true, (patch_height, patch_width), (patch_height, patch_width)
    )  # N x 1 x P x 5 x 5
    dep_patch = split_into_patches(
        dep_feat, (patch_height, patch_width), (patch_height, patch_width)
    )  # N x C x P x 5 x 5

    # Discard patches that have a panoptic value below 0 (ignore) or that all have the same class (no panoptic contours)
    # patch_min = reduce(seg_patch, "n c p h w -> n c p () ()", "min")
    # patch_max = reduce(seg_patch, "n c p h w -> n c p () ()", "max")
    patch_min = seg_patch.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    patch_max = seg_patch.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

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
    mask_pos = ((seg_patch == target_anchor) & patch_valid).int()  # N x C x P x 5 x 5
    mask_neg = ((seg_patch != target_anchor) & patch_valid).int()  # N x C x P x 5 x 5

    # Calculate filtered output (depth) values given the mask
    output_pos = dep_patch * mask_pos  # N x C x P x 5 x 5
    output_neg = dep_patch * mask_neg  # N x C x P x 5 x 5

    # Estimate number of positive and negative features
    target_pos_num = mask_pos.count_nonzero(dim=(-2, -1)).float()  # N x C x P x 1 x 1
    target_neg_num = mask_neg.count_nonzero(dim=(-2, -1)).float()  # N x C x P x 1 x 1

    # Distance terms for all patches
    distance_pos = torch.norm(
        output_anchor * mask_pos - output_pos, p=2, dim=(-2, -1)
    ) / target_pos_num.clamp(1)
    distance_neg = torch.norm(
        output_anchor * mask_neg - output_neg, p=2, dim=(-2, -1)
    ) / target_neg_num.clamp(1)

    # Total loss for all patches
    # patch_losses = torch.maximum(
    #     torch.zeros([distance_pos.shape[0], distance_pos.shape[1], 1], device=seg_true.device),
    #     distance_pos + margin - distance_neg,
    # )  # N x C x P

    patch_losses = distance_pos + margin - distance_neg
    patch_losses = torch.where(patch_losses >= 0, patch_losses, 0)

    # Get mask of patches with positive features and negative features larger than threshold and get filtered patch losses
    mask_pos = target_pos_num > threshold  # N x C x P
    mask_neg = target_neg_num > threshold  # N x C x P
    mask = mask_pos & mask_neg  # N x C x P

    return patch_losses, mask
