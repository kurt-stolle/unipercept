r"""
Implements the kernel mapper module, which takes maps the multipurpose kernel $k^\star$ to all the different
specific kernels using a simple dictionary of heads, represented as a `nn.ModuleDict`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing_extensions import override

from unipercept.nn.layers.norm import GlobalResponseNorm
from unipercept.nn.typings import Activation, Norm

__all__ = ["MapMLP"]


class MapMLP(nn.Module):
    """Straightforward MLP that maps the multipurpose kernel to a task-specific kernel."""

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
    ):
        super().__init__()

        if isinstance(hidden_channels, float):
            hidden_channels = int(in_channels * hidden_channels)
        elif isinstance(hidden_channels, int):
            pass
        else:
            raise ValueError(f"Invalid type for `hidden_channels`: {type(hidden_channels)}")

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = activation()
        self.drop1 = nn.Dropout(dropout)
        self.norm = norm(hidden_channels) if norm is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.drop2 = nn.Dropout(dropout)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop1(y)
        y = self.norm(y)
        y = self.fc2(y)
        y = self.drop2(y)

        return y
