"""
Implements deformable convolutions.

Based on the Torchvision implementation.
"""

from __future__ import annotations

import math
import typing as T

import torch
import torch.fx
import typing_extensions as TX
from torch import Tensor, nn
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torchvision.ops import deform_conv2d

from ._extended import Separable2d
from .utils import NormActivationMixin, to_2tuple

__all__ = ["ModDeform2d", "DeformConv2d"]


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

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def _conv_forward(self, input: Tensor, offset: Tensor, mask: Tensor | None):
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
    def forward(
        self, input: Tensor, offset: Tensor, mask: Tensor | None = None
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
        return self._conv_forward(input, offset, mask)

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


def mask_sigmoid(mask: Tensor, scale: float = 2.0) -> Tensor:
    r"""
    Applies scaled sigmoid activation on mask. The default scale is set to 2 so that initial
    values when ``conv_mask`` is initialized with zeros is 1.

    Parameters
    ----------
    mask: Tensor[batch_size, mask_groups * kernel_area, *out_shape]
        Modulation masks to be multiplied with each output of convolution kernel.
    scale: float
        Positive scaling of the activation output.
    """
    if scale <= 0:
        msg = f"scale must be positive. Got scale={scale}"
        raise ValueError(msg)
    return torch.sigmoid(mask) * scale


class MaskSigmoid(nn.Module):
    """
    See :func:`mask_sigmoid`.
    """

    def __init__(self, scale: float = 2.0):
        super().__init__()
        self.scale = float(scale)

    @TX.override
    def forward(self, mask: Tensor) -> Tensor:
        return mask_sigmoid(mask, self.scale)


def mask_softmax2d(mask: Tensor, kernel_size: tuple[int, int]) -> Tensor:
    r"""
    Performs 2D Mask Softmax Normalization.

    Parameters
    ----------
    mask: Tensor[batch_size, mask_groups * kernel_height * kernel_width,
            out_height, out_width]
        Modulation masks to be multiplied with each output of convolution kernel.
    kernel_size: int or Tuple[int, int]
        Convolution kernel size.

    See Also
    --------
    - Paper: https://arxiv.org/abs/2211.05778


    """
    batch_size, _, out_height, out_width = mask.size()

    weight_h, weight_w = _pair(kernel_size)
    mask_groups = mask.size(1) // (weight_h * weight_w)

    if mask_groups == 0:
        msg = (
            "mask_softmax2d expects the second dimension of mask to be divisible by "
            "kernel_height * kernel_width. Got mask.size(1)={} and kernel_size={}"
        )
        raise RuntimeError(msg)

    mask = mask.view(
        batch_size, mask_groups, weight_h * weight_w, out_height, out_width
    )
    mask = torch.softmax(mask, dim=2)
    mask = mask.view(
        batch_size, mask_groups * weight_h * weight_w, out_height, out_width
    )
    return mask


class MaskSoftmax2d(nn.Module):
    """
    See :func:`mask_softmax2d`
    """

    def __init__(
        self, kernel_size: tuple[int, int] | list[int] | int, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.kernel_size = to_2tuple(kernel_size)

    @TX.override
    def forward(
        self, mask: Tensor, kernel_size: tuple[int, int] | None = None
    ) -> Tensor:
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        return mask_softmax2d(mask, kernel_size)  # type: ignore[arg-type]


class ModDeform2d(NormActivationMixin, DeformConv2d):
    """
    Modulated deformable convolution.
    The offset mask is computed by  a convolutional layer.
    """

    def __init__(
        self,
        *args,
        offset_bias: bool = False,
        mask_bias: bool = False,
        mask_activation: nn.Module | T.Literal["sigmoid", "softmax"] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.offset_generator = Separable2d(
            self.in_channels,
            2 * self.kernel_size[0] * self.kernel_size[1] * self.groups,
            kernel_size=self.kernel_size,  # type: ignore[arg-type]
            stride=self.stride,  # type: ignore[arg-type]
            padding=self.padding,
            dilation=self.dilation,  # type: ignore[arg-type]
            bias=offset_bias,
        )
        self.offset_generator.zero_fill_()
        self.offset_activation = None

        self.mask_generator = Separable2d(
            self.in_channels,
            self.kernel_size[0] * self.kernel_size[1] * self.groups,
            kernel_size=self.kernel_size,  # type: ignore[arg-type]
            stride=self.stride,  # type: ignore[arg-type]
            padding=self.padding,
            dilation=self.dilation,  # type: ignore[arg-type]
            bias=mask_bias,
        )
        self.mask_generator.zero_fill_()
        if mask_activation == "sigmoid":
            self.mask_activation = MaskSigmoid(2.0)
        elif mask_activation == "softmax":
            self.mask_activation = MaskSoftmax2d(self.kernel_size)
        else:
            self.mask_activation = mask_activation or nn.Identity()

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        offset = self.offset_generator(input)
        if self.offset_activation is not None:
            offset = self.offset_activation(offset)

        mask = self.mask_generator(input)
        if self.mask_activation is not None:
            mask = self.mask_activation(mask)

        return self._conv_forward(input, offset, mask)
