r"""
Implements the kernel mapper module, which takes maps the multipurpose kernel $k^\star$ to all the different
specific kernels using a simple dictionary of heads, represented as a `nn.ModuleDict`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing_extensions import override

from .weight import init_trunc_normal_

__all__ = ["MapMLP"]


class MapMLP(nn.Module):
    """Straightforward MLP that maps the multipurpose kernel to a task-specific kernel."""

    def __init__(self, *, in_channels: int, out_channels: int, ratio=1.0):
        super().__init__()

        mid_channels = int(in_channels * ratio)

        self.fc1 = nn.Linear(in_channels, mid_channels, bias=True)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(mid_channels, eps=1e-6)
        self.fc2 = nn.Linear(mid_channels, out_channels, bias=False)

        self.fc1.apply(init_trunc_normal_)
        self.fc2.apply(init_trunc_normal_)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.norm(y)
        y = self.fc2(y)

        return y
