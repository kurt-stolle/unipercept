"""Implements modules for incorporating coord information into a layer, a la CoordCat."""

from __future__ import annotations

import math
import typing as T
from typing import override

import torch
import torch.fx
from torch import Tensor, nn
from torch.nn.functional import interpolate

__all__ = ["CoordModuleProtocol", "CoordCat2d", "CoordEmbed2d", "CoordSinusoid2d"]


class CoordModuleProtocol(T.Protocol):
    """
    Protocol for a module that adds positional encoding to an input tensor.
    """

    cat_channels: T.Final[int]

    def forward(self, t: Tensor) -> Tensor: ...

    if T.TYPE_CHECKING:
        __call__ = forward


class CoordCat2d(nn.Module):
    """
    Layer that concatenates a 2D coordinate grid to the input tensor.
    """

    gamma: T.Final[float]
    groups: T.Final[int]
    cat_channels: T.Final[int]

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

        # Concatenate the groups back together
        return _make_grid(t_split, self.gamma, t.shape, t.device)

    if T.TYPE_CHECKING:
        __call__ = forward


class CoordEmbed2d(nn.Module):
    """
    Layer that embeds a learnable position encoding to the input tensor.
    The encoding is interpolated to match the size of the input tensor.
    """

    cat_channels: T.Final[int]

    def __init__(self, channels: int = 1, height: int = 32, aspect_ratio: int = 2):
        """
        Parameters
        ----------
        channels : int
            The number of channels of the emebdding, by default 1.
        dims : int
            The number of dimensions of the embedding (height * width).
        aspect_ratio : int
            The aspect ratio of the embedding (width / height). Default is 2.
        """
        super().__init__()
        self.cat_channels = 0
        self.embedding = nn.Parameter(
            torch.randn(channels, height, height * aspect_ratio)
        )

    @override
    def forward(self, t: Tensor) -> Tensor:
        e = self.embedding.unsqueeze(0).expand(t.shape[0], -1, -1, -1)
        e = interpolate(
            e.unsqueeze(0), size=t.shape[1:], mode="trilinear", align_corners=False
        ).squeeze(0)

        return t + e

    if T.TYPE_CHECKING:
        __call__ = forward


class CoordSinusoid2d(nn.Module):
    cat_channels: T.Final[int]
    cache: dict[tuple[int, int, int, torch.device], Tensor]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cat_channels = 0
        self.cache = {}

    def _cached_sinusoidal_encoding(
        self, c: int, h: int, w: int, d: torch.device
    ) -> Tensor:
        key = (c, h, w, d)
        if key not in self.cache:
            self.cache[key] = _make_sinusoidal_encoding(c, h, w, d)
        return self.cache[key]

    @override
    def forward(self, x: Tensor) -> Tensor:
        # if torch.compiler.is_dynamo_compiling():
        #    enc_fn = _make_sinusoidal_encoding
        # else:
        #    enc_fn = self._cached_sinusoidal_encoding
        enc_fn = self._cached_sinusoidal_encoding
        return x + enc_fn(x.size(1), x.size(2), x.size(3), x.device)

    if T.TYPE_CHECKING:
        __call__ = forward


def _make_grid(
    t_split: torch.Tensor, gamma: float, shape: torch.Size, device: torch.device
) -> torch.Tensor:
    with torch.no_grad():
        x_pos = torch.linspace(-gamma, gamma, shape[-2], device=device)
        y_pos = torch.linspace(-gamma, gamma, shape[-1], device=device)

        grid_x, grid_y = torch.meshgrid(x_pos, y_pos, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(shape[0], -1, -1, -1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(shape[0], -1, -1, -1)
    t_list = [torch.cat([t_n, grid_x, grid_y], dim=1) for t_n in t_split]
    return torch.cat(t_list, dim=1)


def _make_sinusoidal_encoding(
    channels: int, height: int, width: int, device: torch.device
) -> Tensor:
    assert channels % 4 == 0, "Channels must be divisible by 4 for sinusoidal encoding"

    with device, torch.no_grad():
        y_position = torch.arange(0, height).unsqueeze(1)
        x_position = torch.arange(0, width).unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, channels, 4) * -(math.log(10000.0) / channels)
        )
        div_term = div_term.view(-1, 1, 1)
        pe = torch.zeros(channels, height, width)
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

    return pe


def _get_split_size(t: Tensor, groups: int) -> int:
    return int(t.size(1)) // groups


torch.fx.wrap("_make_grid")
torch.fx.wrap("_get_split_size")
