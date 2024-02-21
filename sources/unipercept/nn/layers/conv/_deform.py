"""
Implements deformable convolutions.

Based on the Torchvision implementation. 
"""

from __future__ import annotations

import math
import typing as T
from typing import Optional, Tuple

import torch
import torch.nn as nn
import typing_extensions as TX
from torch import Tensor, nn
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

from ..weight import init_xavier_fill_
from ._extended import Conv2d
from .utils import NormActivationMixin, PaddingMixin

__all__ = ["ModDeform2d", "DeformConv2d"]


@torch.fx.wrap
def torch_zeros(*args, **kwargs):
    return torch.zeros(*args, **kwargs)


def deform_conv2d(
    input: Tensor,
    offset: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Performs Deformable Convolution v2, described in
    `Deformable ConvNets v2: More Deformable, Better Results
    <https://arxiv.org/abs/1811.11168>`__ if :attr:`mask` is not ``None`` and
    Performs Deformable Convolution, described in
    `Deformable Convolutional Networks
    <https://arxiv.org/abs/1703.06211>`__ if :attr:`mask` is ``None``.


    Parameters
    ----------
    input: Tensor[batch_size, in_channels, in_height, in_width]
        Input tensor
    offset: Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]
        Output tensor offsets to be applied for each position in the convolution kernel.
    weight: Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]
        Convolution weights split into groups of size (in_channels // groups)
    bias: Tensor[out_channels]
        Optional bias of shape (out_channels,). Default: None
    stride:
        Distance between convolution centers. Default: 1
    padding:
        Height/width of padding of zeroes around each image. Default: 0
    dilation:
        Spacing between kernel elements. Default: 1
    mask: Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]
        Masks to be applied for each position in the convolution kernel. Default: None

    Returns
    -------
        Tensor[batch_sz, out_channels, out_h, out_w]: result of convolution

    Examples
    --------

    >>> input = torch.rand(4, 3, 10, 10)
    >>> kh, kw = 3, 3
    >>> weight = torch.rand(5, 3, kh, kw)
    >>> # offset and mask should have the same spatial size as the output
    >>> # of the convolution. In this case, for an input of 10, stride of 1
    >>> # and kernel size of 3, without padding, the output size is 8
    >>> offset = torch.rand(4, 2 * kh * kw, 8, 8)
    >>> mask = torch.rand(4, kh * kw, 8, 8)
    >>> out = deform_conv2d(input, offset, weight, mask=mask)
    >>> print(out.shape)
    >>> # returns
    >>>  torch.Size([4, 5, 8, 8])
    """
    out_channels = weight.shape[0]

    use_mask = mask is not None

    if mask is None:
        mask = torch_zeros((input.shape[0], 1), device=input.device, dtype=input.dtype)

    if bias is None:
        bias = torch_zeros(out_channels, device=input.device, dtype=input.dtype)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)
    weights_h, weights_w = weight.shape[-2:]
    _, n_in_channels, _, _ = input.shape

    n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
    n_weight_grps = n_in_channels // weight.shape[1]

    return torch.ops.torchvision.deform_conv2d(
        input,
        weight,
        offset,
        mask,
        bias,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dil_h,
        dil_w,
        n_weight_grps,
        n_offset_grps,
        use_mask,
    )


class DeformConv2d(nn.Module):
    """
    See :func:`deform_conv2d`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = Parameter(
            torch.empty(
                out_channels,
                in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1],
            )
        )

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    @TX.override
    def forward(
        self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Parameters
        ----------
        input: Tensor[batch_size, in_channels, in_height, in_width])
            Input tensor
        offset: Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]
            Offsets to be applied for each position in the convolution kernel.
        mask: Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]
            Masks to be applied for each position in the convolution kernel.
        """
        return deform_conv2d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    @TX.override
    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", kernel_size={self.kernel_size}"
            f", stride={self.stride}"
        )
        s += f", padding={self.padding}" if self.padding != (0, 0) else ""
        s += f", dilation={self.dilation}" if self.dilation != (1, 1) else ""
        s += f", groups={self.groups}" if self.groups != 1 else ""
        s += ", bias=False" if self.bias is None else ""
        s += ")"

        return s


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

        self.mod.apply(init_xavier_fill_)

    @TX.override
    def forward(self, input):
        x = self.mod(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.deform(input, offset, mask)
        return output
