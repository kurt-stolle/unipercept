from __future__ import annotations

import torch
from unicore.utils.tensorclass import Tensorclass


class CameraModel(Tensorclass):
    """
    Build pinhole camera calibration matrices.

    See: https://kornia.readthedocs.io/en/latest/geometry.camera.pinhole.html
    """

    image_size: torch.Tensor  # shape: ..., 2 (height, width)
    matrix: torch.Tensor  # shape: (... x 4 x 4) K
    pose: torch.Tensor  # shape: (... x 4 x 4) Rt

    @property
    def height(self) -> torch.Tensor:
        return self.image_size[..., 0]

    @property
    def width(self) -> torch.Tensor:
        return self.image_size[..., 1]

    def __post_init__(self):
        if self.matrix.shape[-2:] != (4, 4):
            raise ValueError("Camera matrix must be of shape (..., 4, 4)")
        if self.pose.shape[-2:] != (4, 4):
            raise ValueError("Camera pose must be of shape (..., 4, 4)")
        if self.image_size.shape[-1] != 2:
            raise ValueError("Camera size must be of shape (..., 2)")
