"""
Implements a module that normalizes input captures.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
import unipercept.data.points
from typing_extensions import override

if T.TYPE_CHECKING:
    from torch.types import Device

__all__ = ["Normalizer"]


class Normalizer(nn.Module):
    """
    Normalizes input captures
    """

    def __init__(self, mean: list[float], std: list[float], image_format: str | None = None):
        super().__init__()

        self.image_format = image_format
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1), False)
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1), False)
        assert self.mean.shape == self.std.shape, f"{self.mean} and {self.std} have different shapes!"

    @property
    def device(self) -> Device:
        return self.mean.device

    @override
    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Denormalize an image to a float with values [0, 1]
        """
        image = image.to(device=self.device, dtype=torch.float32) * self.std + self.mean

        if self.image_format == "BGR":
            image = image[[2, 1, 0], :, :]
        elif self.image_format != "RGB":
            raise ValueError(f"Unknown image format: {self.image_format}")

        image /= 255.0
        image = image.clamp(0, 1)

        return image

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize an image.
        """
        return (image.to(device=self.device) - self.mean) / self.std

    def forward(self, data: unipercept.data.points.InputData) -> unipercept.data.points.InputData:
        """
        Copy data to device and normalize each captures input image.
        """

        data.captures.images = self.normalize(data.captures.images).as_subclass(unipercept.data.points.Image)

        return data
