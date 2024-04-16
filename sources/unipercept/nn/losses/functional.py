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
        # strides = (sizes[0], sizes[1])
        strides = (1, 1)

    batch_size, channels, _, _ = x.shape

    for dim, (size, stride) in enumerate(zip(sizes, strides)):
        x = x.unfold(dim + 2, size, stride)

    # B x C x P x H x W
    return x.reshape(batch_size, channels, -1, sizes[0], sizes[1])


def depth_to_normals(depth: torch.Tensor, fx: float, fy: float) -> torch.Tensor:
    r"""
    Compute surface normals from depth map.
    """
    # Compute gradients in x and y directions
    depth_dx = torch.nn.functional.conv2d(
        depth, torch.tensor([[[[0, 0, 0], [-1, 0, 1], [0, 0, 0]]]]).to(depth.device)
    )
    depth_dy = torch.nn.functional.conv2d(
        depth, torch.tensor([[[[0, -1, 0], [0, 0, 0], [0, 1, 0]]]]).to(depth.device)
    )

    # Compute surface normals
    normals = torch.stack(
        [
            -depth_dx * fx,
            -depth_dy * fy,
            torch.ones_like(depth),
        ],
        dim=-1,
    )

    # Normalize normals
    normals = torch.nn.functional.normalize(normals, p=2, dim=-1)

    return normals


def depth_to_disp(
    depth: torch.Tensor,
    depth_min: float,
    depth_max: float,
    fx: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""
    Convert depth map to disparity map.
    """
    return torch.where(depth > eps, fx / depth, torch.zeros_like(depth))


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

