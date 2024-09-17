"""
Implements segmentation-to-depth and vice-versa losses.
"""

from __future__ import annotations

import typing as T
from typing import override

import torch
from einops import rearrange
from torch import Tensor, nn

from unipercept.data.tensors import absolute_to_normalized_depth
from unipercept.nn.losses.functional import split_into_patches
from unipercept.nn.losses.mixins import ScaledLossMixin, StableLossMixin


def depth_guided_segmentation_loss(
    seg_feat: Tensor,
    dep_true: Tensor,
    tau: float,
    patch_size: tuple[int, int],
    patch_stride: tuple[int, int],
    valid_ratio: float = 0.5,
    eps: float = 1e-8,
) -> Tensor:
    r"""
    Compute the depth-guided segmentation loss.

    Parameters
    ----------
    seg_feat : Tensor[B, C, H, W]
        The segmentation features. No normalization is applied, depending on the
        initialization of the feature space, it could be beneficial to normalize
        the features in the channel dimension before passing them to this function.
    dep_true : Tensor[B, H, W]
        The ground truth depth map. No normalization is applied, it is recommended to
        tune the ``tau`` parameter to the range of the depth values.
    tau : int
        The temperature parameter for the depth loss.
    patch_size : Tuple[int, int]
        The size of the patch to use. If the size of the depth map is larger than
        the segmentation map, then the patch size will be scaled accordingly and
        median pooling is applied such that the depth map matches the segmentation map.
    patch_stride : Tuple[int, int]
        The stride of the patch.
    valid_ratio : float
        The ratio of depth values that must be valid within a patch to have it be
        a valid patch.
    eps : float
        A small value to avoid division by zero.
    """
    assert dep_true.ndim == 3, f"Expected B x H x W, got: {dep_true.shape}"
    assert seg_feat.ndim == 4, f"Expected B x C x H x W, got: {seg_feat.shape}"

    # Ground truth processing
    with torch.no_grad():
        # Compute the the scale of the patch size and stride such that an equal amount
        # of patches are generated
        W_seg, H_seg = seg_feat.shape[-2:]
        W_dep, H_dep = dep_true.shape[-2:]
        scale_x = W_dep // W_seg
        scale_y = H_dep // H_seg
        dep_patch = split_into_patches(
            dep_true,
            (int(patch_size[0] * scale_x), int(patch_size[1] * scale_y)),
            (int(patch_stride[0] * scale_x), int(patch_stride[1] * scale_y)),
        )

        # Downsample patches via median pooling to match the segmentation size
        dep_patch = rearrange(
            dep_patch,
            "n p (h pH) (w pW) -> n p h w (pH pW)",
            h=patch_size[0],
            w=patch_size[1],
        )
        dep_valid = dep_patch > eps
        dep_patch[~dep_valid] = torch.nan
        dep_patch = torch.nanmedian(dep_patch, dim=-1).values
        dep_valid = dep_valid.int().sum(dim=-1) / dep_valid.shape[-1] > valid_ratio
        dep_patch[~dep_valid] = 0.0

        c_x = patch_size[0] // 2
        c_y = patch_size[1] // 2
        center_mask = torch.zeros(patch_size, device=dep_patch.device, dtype=torch.bool)
        center_mask[c_x, c_y] = True

        dep_anchor = dep_patch[..., center_mask]
        dep_patch = dep_patch[..., ~center_mask]
        dep_diff = (dep_anchor - dep_patch).abs()

        valid_diff = (dep_patch > 0) & (dep_anchor > 0)  # (B, P, pH * pW)
        valid_amount = valid_diff.float().sum(dim=(-2, -1))  # (B)

    # if not valid_amount.any():
    #    return seg_feat * 0.0

    # Segmentation features
    seg_patch = split_into_patches(seg_feat, patch_size, patch_stride)
    seg_center = seg_patch[..., center_mask]
    seg_patch = seg_patch[..., ~center_mask]
    seg_diff = torch.linalg.vector_norm((seg_center - seg_patch), dim=1, ord=2.0)

    # seg_center = seg_patch[..., c_x : c_x + 1, c_y : c_y + 1]  # .contiguous()
    # seg_diff = torch.norm(seg_center - seg_patch, dim=1)  # .clamp(min=eps)

    # valid_diff = valid_diff & (seg_diff > eps)

    loss = torch.exp(-dep_diff / tau) * torch.exp(-(seg_diff**2))  # (B, P, pH * pW)
    loss = loss.sum(dim=(-2, -1)) / valid_amount.clamp(min=1)  # (B)

    return loss


class DGPLoss(StableLossMixin, ScaledLossMixin, nn.Module):
    """
    Implements a depth-guided panoptic loss (DGP) loss
    """

    tau: T.Final[float]
    min_depth: T.Final[float]
    max_depth: T.Final[float]
    mode: T.Final[str]
    patch_size: T.Final[tuple[int, int]]
    patch_stride: T.Final[tuple[int, int]]

    def __init__(
        self,
        *,
        tau=1.0,
        min_depth: float = 1.0,
        max_depth: float = 100.0,
        mode: T.Literal["absolute", "disparity"] = "disparity",
        patch_size: tuple[int, int] = (5, 5),
        patch_stride: tuple[int, int] = (1, 1),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tau = tau
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.mode = mode

    @classmethod
    def from_metadata(cls, dataset_name: str, **kwargs) -> T.Self:
        from unipercept.data.sets import catalog

        info = catalog.get_info(dataset_name)
        assert info is not None

        return cls(
            min_depth=info.depth_min,
            max_depth=info.depth_max,
            **kwargs,
        )

    @override
    def forward(self, seg_feat: Tensor, dep_true: Tensor):
        # Normalize depth values to the range [0, 1]
        with torch.no_grad():
            dep_true = absolute_to_normalized_depth(
                dep_true, self.min_depth, self.max_depth, self.mode
            )

        # Normalize each channel of the segmentation feature space
        seg_feat = nn.functional.normalize(seg_feat, dim=1, p=2.0)

        # Compute loss
        loss = depth_guided_segmentation_loss(
            seg_feat,
            dep_true,
            self.tau,
            self.patch_size,
            self.patch_stride,
            self.eps,
        )
        return loss.mean() * self.scale


def segmentation_guided_triplet_loss(
    dep_feat: Tensor,
    seg_true: Tensor,
    margin: float,
    threshold: int,
    patch_size: tuple[int, int],
    patch_stride: tuple[int, int],
) -> Tensor:
    if seg_true.ndim == dep_feat.ndim - 1:
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
        seg_patch = split_into_patches(
            seg_true, patch_size, patch_stride
        )  # B x 1 x P x 5 x 5

        c_x = patch_size[0] // 2
        c_y = patch_size[1] // 2
        center_mask = torch.zeros(patch_size, device=seg_patch.device, dtype=torch.bool)
        center_mask[c_x, c_y] = True

        tgt_anchor = seg_patch[..., center_mask]  # (B, 1, P, 1)
        tgt_values = seg_patch[..., ~center_mask]  # (B, 1, P, pH * pW - 1 = N)

        mask_same = (tgt_anchor == tgt_values).any(dim=-1, keepdim=True)
        mask_anchor = tgt_anchor >= 0
        mask_values = tgt_values >= 0
        mask_valid = (mask_same & mask_anchor).expand_as(mask_values) & mask_values

        w_pos = ((tgt_values == tgt_anchor) & mask_valid).float()
        n_pos = w_pos.sum(dim=-1)

        w_neg = ((tgt_values != tgt_anchor) & mask_valid).float()
        n_neg = w_neg.sum(dim=-1)

        mask_threshold = (n_pos > threshold) & (n_neg > threshold)  # (B, 1, P)
        w_pos[~mask_threshold].zero_()
        w_neg[~mask_threshold].zero_()

    # Split depth features into patches
    dep_patch = split_into_patches(
        dep_feat, patch_size, patch_stride
    )  # B x C x P x 5 x 5
    src_anchor = dep_patch[..., center_mask]  # (B, C, P, 1)
    src_values = dep_patch[..., ~center_mask]  # (B, C, P, N)

    # Expand the anchors such that they have the same shape as the values
    # src_anchor = src_anchor.expand_as(src_values)  # (B, C, P, N)
    src_pos = src_values * w_pos  # (N, C, P, N)
    src_neg = src_values * w_neg

    # Compute the distance (l2) between the anchor and the positive/negative features
    def _distance_to_anchor(a: Tensor, b: Tensor) -> Tensor:
        d = a - b  # a * w - b  # (B, C, P, N)
        d = torch.linalg.vector_norm(d, dim=1, ord=2.0, keepdim=True)  # (B, 1, P, N)
        # d = d.sum(dim=-1) / w.sum(dim=-1).clamp_min(1.0)  # (B, 1, P)
        return d

    d_pos = _distance_to_anchor(src_anchor, src_pos)
    d_neg = _distance_to_anchor(src_anchor, src_neg)
    # d_neg = torch.norm(
    #    src_anchor * weights_neg - src_neg, p=2, dim=-1
    # ) / n_neg.clamp_min(1.0)

    # Total loss for all patches
    # loss = d_pos + margin - d_neg  # (B, 1, P, N)
    loss = ((d_pos - d_neg).abs() - margin).relu()  # (B, 1, P)

    # Take the weighted average per batch item, then the mean over all batches
    w_total = w_pos + w_neg
    loss = loss.flatten(1).sum(dim=-1) / w_total.flatten(1).sum(dim=-1).clamp_min(1.0)

    return loss


class PGTLoss(StableLossMixin, ScaledLossMixin, nn.Module):
    """
    Panoptic-guided Triplet Loss (PGT) loss

    Paper: https://arxiv.org/abs/2210.07577
    """

    patch_size: T.Final[tuple[int, int]]
    patch_stride: T.Final[tuple[int, int]]
    margin: T.Final[float]
    threshold: T.Final[int]

    def __init__(
        self,
        *,
        patch_size: tuple[int, int] = (5, 5),
        patch_stride: tuple[int, int] | None = (1, 1),
        margin=1e-2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.patch_stride = patch_stride or patch_size
        self.margin = margin
        self.threshold = max(1, min(*patch_size) // 2)

    @override
    def forward(self, dep_feat: Tensor, seg_true: Tensor):
        dep_feat = nn.functional.normalize(dep_feat, dim=1, p=2.0)
        loss = segmentation_guided_triplet_loss(
            dep_feat,
            seg_true,
            self.margin,
            self.threshold,
            self.patch_size,
            self.patch_stride,
        )
        return loss.mean() * self.scale


class PGSLoss(ScaledLossMixin, nn.Module):
    """
    Panoptic-guided Smoothness Loss (PGS)

    Paper: https://arxiv.org/abs/2210.07577
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @override
    def forward(self, disparity: Tensor, panoptic: Tensor):
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
