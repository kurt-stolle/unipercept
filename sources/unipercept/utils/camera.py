from __future__ import annotations

import torch
from torch import Tensor


def build_calibration_matrix(
    focal_lengths: list[tuple[float, float]] | Tensor,
    principal_points: list[tuple[float, float]] | Tensor,
    orthographic: bool,
) -> Tensor:
    """
    Build the calibration matrix (K-matrix) using the focal lengths and
    principal points.
    """

    # Focal length, shape (), (N, 1) or (N, 2)
    if not isinstance(focal_lengths, Tensor):
        focal_lengths = torch.tensor([list(t) for t in focal_lengths])
    if focal_lengths.ndim in [0, 1] or focal_lengths.shape[1] == 1:
        fx = fy = focal_lengths
    else:
        fx, fy = focal_lengths.unbind(1)

    # Principal point
    if not isinstance(principal_points, Tensor):
        principal_points = torch.tensor([list(p) for p in principal_points])
    px, py = principal_points.unbind(1)

    # Amount of cameras
    N = len(focal_lengths)

    # Value checks
    assert N == len(principal_points)
    assert focal_lengths.device == principal_points.device

    # Create calibration matrix
    K = fx.new_zeros(N, 4, 4)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    if orthographic:
        K[:, 0, 3] = px
        K[:, 1, 3] = py
        K[:, 2, 2] = 1.0
        K[:, 3, 3] = 1.0
    else:
        K[:, 0, 2] = px
        K[:, 1, 2] = py
        K[:, 3, 2] = 1.0
        K[:, 2, 3] = 1.0

    return K
