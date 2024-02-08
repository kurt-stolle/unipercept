"""
Implements a module that normalizes input captures.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing_extensions import override

__all__ = ["Normalizer"]


class Normalizer(nn.Module):
    """
    Normalizes input captures
    """

    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()

        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1))
        assert (
            self.mean.shape == self.std.shape
        ), f"{self.mean} and {self.std} have different shapes!"

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize an image.
        """
        return (image - self.mean) / self.std  # type: ignore

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Copy data to device and normalize each captures input image.
        """

        return self.normalize(x)
