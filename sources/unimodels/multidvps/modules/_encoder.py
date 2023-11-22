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
from unipercept.nn.layers.norm import GroupNormCG
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
            self.coord = nn.Identity()

        # Supported conv modules have the same signature.
        if deform:
            conv_module = ModDeform2d.with_norm_activation
        else:
            conv_module = Conv2d.with_norm_activation

        # Create `num_convs` conv modules.
        self.layers = nn.ModuleList()
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
                padding=1,
                stride=1,
                groups=cur_groups,
                bias=norm is None,
                norm=norm,
                activation=activation,
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.coord(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x
