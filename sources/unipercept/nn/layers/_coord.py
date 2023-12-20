"""Implements modules for incorporating coord information into a layer, a la CoordCat."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import override

__all__ = ["CoordCat2d"]


class CoordCat2d(nn.Module):
    """Layer that concatenates a 2D coordinate grid to the input tensor."""

    def __init__(self, groups: int = 1, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.groups = groups
        self.cat_channels = 2 * groups

        assert self.gamma > 0.0, f"Gamma must be positive: {gamma}"

    @staticmethod
    # @lru_cache(maxsize=16)
    def _make_grid(gamma: float, shape: torch.Size, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_pos = torch.linspace(-gamma, gamma, shape[-2], device=device)
            y_pos = torch.linspace(-gamma, gamma, shape[-1], device=device)

            grid_x, grid_y = (
                g.unsqueeze_(0).unsqueeze_(0).expand(shape[0], -1, -1, -1)
                for g in torch.meshgrid(x_pos, y_pos, indexing="ij")
            )
            del x_pos, y_pos
        return grid_x, grid_y

    @override
    def forward(self, t: Tensor) -> Tensor:
        # Split the tensor into groups
        t_split = torch.split(t, t.shape[1] // self.groups, dim=1)

        # Add the grid to each group
        grid_x, grid_y = self._make_grid(self.gamma, t.shape, t.device)
        t_split = [torch.cat([t_n, grid_x, grid_y], dim=1) for t_n in t_split]

        # Concatenate the groups back together
        return torch.cat(t_split, dim=1)
