"""Implements structures for the camera intrinsic and extrinsic parameters."""

from __future__ import annotations

import torch

__all__ = ["build_pinhole_intrinsics", "build_pinhole_extrinsics"]


def build_pinhole_intrinsics(fx, fy, u0, v0) -> torch.Tensor:
    K = torch.zeros((4, 4), dtype=torch.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = u0
    K[1, 2] = v0
    K[3, 2] = 1.0
    K[2, 3] = 1.0

    return K


def build_pinhole_extrinsics(
    yaw: float,
    pitch: float,
    roll: float,
    x: float,
    y: float,
    z: float,
) -> torch.Tensor:
    from math import cos, sin

    # Rotation I -> C (image to camera)
    R_ic = torch.tensor([[0, -1, 1], [0, 0, -1], [1, 0, 0]], dtype=torch.float32)

    # Rotation C -> V (camera to vehicle)
    s_y, s_p, s_r = map(sin, (yaw, pitch, roll))
    c_y, c_p, c_r = map(cos, (yaw, pitch, roll))

    R_cv = torch.tensor(
        [
            [c_y * c_p, c_y * s_p * s_r - s_y * c_r, c_y * s_p * c_r + s_y * s_r],
            [s_y * c_p, s_y * s_p * s_r + c_y * c_r, s_y * s_p * c_r - c_y * s_r],
            [-s_p, c_p * s_r, c_p * c_r],
        ],
        dtype=torch.float32,
    )

    R = R_ic @ R_cv

    # Translation C -> V (camera to vehicle)
    t_cv = torch.tensor([x, y, z], dtype=torch.float32)

    # Extrinsics matrix [R|t]
    Rt = torch.zeros((4, 4), dtype=torch.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = t_cv
    Rt[3, 3] = 1.0

    return Rt
