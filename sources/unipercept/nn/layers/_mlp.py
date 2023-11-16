"""
Implements the kernel mapper module, which takes maps the multipurpose kernel $k^\star$ to all the different
specific kernels using a simple dictionary of heads, represented as a `nn.ModuleDict`.
"""

import torch
import torch.nn as nn
import typing as T
from typing_extensions import override

from unipercept.nn.typings import Activation, Norm

__all__ = ["MapMLP", "EmbedMLP"]


class MapMLP(nn.Module):
    """Straightforward MLP that maps the multipurpose kernel to a task-specific kernel."""

    eps: T.Final[float]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | float = 1.0,
        *,
        dropout=0.0,
        norm: Norm | None = nn.LayerNorm,
        bias=True,
        activation: Activation = nn.GELU,
        eps: float = 1e-8,
    ):
        super().__init__()

        if isinstance(hidden_channels, float):
            hidden_channels = int(in_channels * hidden_channels)
        elif isinstance(hidden_channels, int):
            pass
        else:
            raise ValueError(f"Invalid type for `hidden_channels`: {type(hidden_channels)}")

        self.eps = eps
        self.fc1 = nn.Linear(in_channels, hidden_channels, bias=bias)
        self.act = activation()
        self.drop1 = nn.Dropout(dropout, inplace=True)
        self.norm = norm(hidden_channels) if norm is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.drop2 = nn.Dropout(dropout, inplace=True)

    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop1(y)
        y = self.norm(y)
        y = self.fc2(y)
        y = self.drop2(y)

        return y

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_mlp(x)


class EmbedMLP(MapMLP):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self._forward_mlp(x), p=2, dim=1, eps=self.eps)
