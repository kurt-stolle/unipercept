"""
Implements a module that normalizes input captures.
"""

import typing as T

import torch
from torch import nn

__all__ = ["Normalizer"]


class Normalizer(nn.Module):
    """
    Normalizes input captures
    """

    def __init__(self, mean: T.Sequence[float], std: T.Sequence[float]):
        super().__init__()

        mean = list(map(float, mean))
        self.register_buffer(
            "mean", torch.tensor(mean, requires_grad=False).view(-1, 1, 1).clone()
        )

        std = list(map(float, std))
        self.register_buffer(
            "std", torch.tensor(std, requires_grad=False).view(-1, 1, 1).clone()
        )

        assert (
            self.mean.shape == self.std.shape
        ), f"{self.mean} and {self.std} have different shapes!"

    @T.override
    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize an image.
        """
        return (image - self.mean) / self.std  # type: ignore

    def extra_repr(self) -> str:
        return f"mean={self.mean.view(-1).tolist()}, std={self.std.view(-1).tolist()}"
