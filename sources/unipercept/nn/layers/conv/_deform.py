"""
Implements deformable convolutions.

Now uses the `torchvision.ops.DeformConv2d` module to do the heavy lifting.
"""

from __future__ import annotations


import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from typing_extensions import override

from ..weight import init_xavier_fill_
from ._extended import Conv2d
from .utils import NormActivationMixin, PaddingMixin

__all__ = ["ModDeform2d"]


class ModDeform2d(NormActivationMixin, PaddingMixin, nn.Module):
    """
    Modulated deformable convolution.
    The offset mask is computed by  a convolutional layer.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding: int | tuple[int, int] = 1,
        dilation=1,
        groups=1,
        bias: bool = True,
    ):
        # padding = self.parse_padding(padding, kernel_size, dilation=dilation, stride=stride)
        super().__init__()

        if isinstance(padding, tuple):
            assert (
                not padding[1] or padding[1] == padding[0]
            ), "Asymmetric padding is not supported"
            padding = padding[0]

        self.mod = Conv2d(
            in_channels,
            groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            bias=False,
        )
        self.deform = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.out_channels = out_channels
        self.in_channels = in_channels

        self.apply(init_xavier_fill_)

    @override
    def forward(self, input):
        x = self.mod(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.deform(input, offset, mask)
        return output
