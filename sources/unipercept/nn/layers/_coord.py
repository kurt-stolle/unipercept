"""Implements modules for incorporating coord information into a layer, a la CoordCat."""

from __future__ import annotations

import typing as T

import torch
import torch.fx
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

__all__ = ["CoordCat2d"]


class CoordCat2d(nn.Module):
    """Layer that concatenates a 2D coordinate grid to the input tensor."""

    gamma: torch.jit.Final[float]
    groups: torch.jit.Final[int]
    cat_channels: torch.jit.Final[int]

    def __init__(self, groups: int = 1, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.groups = groups
        self.cat_channels = 2 * groups

        assert self.gamma > 0.0, f"Gamma must be positive: {gamma}"

    @override
    def forward(self, t: Tensor) -> Tensor:
        # Split the tensor into groups
        split_size: int = _get_split_size(t, self.groups)
        t_split = t.split(split_size, dim=1)

        # Add the grid to each group
        grid_x, grid_y = _make_grid(self.gamma, t.shape, t.device)

        t_split = [torch.cat([t_n, grid_x, grid_y], dim=1) for t_n in t_split]

        # Concatenate the groups back together
        return torch.cat(t_split, dim=1)


@torch.fx.wrap
@torch.no_grad()
def _make_grid(
    gamma: float, shape: torch.Size, device: torch.device
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    x_pos = torch.linspace(-gamma, gamma, shape[-2], device=device)
    y_pos = torch.linspace(-gamma, gamma, shape[-1], device=device)

    grid_x, grid_y = torch.meshgrid(x_pos, y_pos, indexing="ij")
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(shape[0], -1, -1, -1)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(shape[0], -1, -1, -1)

    return grid_x, grid_y


@torch.fx.wrap
def _get_split_size(t: Tensor, groups: int) -> int:
    return int(t.size(1)) // groups
