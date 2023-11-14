"""Imports an encoding block with a coordinate layer (CoordConv) and a downsampling block."""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from typing_extensions import override

from unipercept.nn.layers._coord import CoordCat2d
from unipercept.nn.layers.conv import Conv2d, ModDeform2d, Separable2d
from unipercept.nn.layers.conv.utils import AvgPool2dSame
from unipercept.nn.layers.weight import init_xavier_fill_

if T.TYPE_CHECKING:
    from unipercept.nn.typings import Activation, Norm

__all__ = ["Encoder"]


class Encoder(nn.Module):
    """Block of convs concatenated with a coordinate layer (CoordConv)."""

    in_channels: T.Final[int]
    out_channels: T.Final[int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int,
        deform=False,
        coord: T.Optional[CoordCat2d] = None,
        norm: T.Optional[Norm] = None,
        groups: int = 1,
        activation: T.Optional[Activation] = nn.GELU,
        **kwargs,
    ):
        """Groups and kernel size can be specified as a list for each layer or a single value for all layers."""
        super().__init__(**kwargs)

        # Coordinate conv as specified in the PanopticFCN paper.
        if coord:
            assert isinstance(coord, nn.Module), f"Expected nn.Module, got {type(coord)}!"
            assert hasattr(coord, "cat_channels")
            in_channels += coord.cat_channels
            self.coord = coord
        else:
            self.coord = None

        # Supported conv modules have the same signature.
        if deform:
            conv_module = ModDeform2d.with_norm_activation
        else:
            conv_module = Conv2d.with_norm_activation

        # Create `num_convs` conv modules.
        self.layers = nn.Sequential()
        for n in range(num_convs):
            if n == 0:
                cur_channels = in_channels
                cur_groups = 1
            else:
                cur_channels = out_channels
                cur_groups = groups

            c = conv_module(
                cur_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=cur_groups,
                bias=norm is None,
                norm=norm if norm is not None else nn.Identity,
                activation=activation if activation is not None else nn.Identity,
            )
            self.layers.add_module(f"conv_{n}", c)

            # Channel shuffle if groups > 0 to prevent information loss by
            # vanishing gradients.
            if cur_groups > 1 and cur_groups != out_channels:
                shf = Rearrange("b (c1 c2) h w -> b (c2 c1) h w", c1=groups)
                self.layers.add_module(f"shf_{n}", shf)

        # Helper variables for other modules to access the number of input and output channels.
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initialize weights.
        self.apply(init_xavier_fill_)

    @override
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.coord is not None:
            inputs = self.coord(inputs)
        inputs = self.layers(inputs)

        return inputs


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            self.pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
        else:
            self.pool = nn.Identity()

        if in_channels != out_channels:
            self.conv = Conv2d(in_channels, out_channels, 1, stride=1)
        else:
            self.conv = nn.Identity()

    @override
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x
