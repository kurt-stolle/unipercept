from __future__ import annotations

import typing as T

import fvcore.nn.weight_init
import torch
import torch.nn as nn
import typing_extensions as TX

from .utils import NormActivationMixin, PaddingMixin

__all__ = ["Conv2d", "Standard2d", "PadConv2d", "Separable2d"]


class Conv2d(NormActivationMixin, nn.Conv2d):
    pass


class PadConv2d(PaddingMixin, Conv2d):
    @TX.override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._padding_forward(x, self.kernel_size, self.stride, self.dilation)
        x = self._conv_forward(x, self.weight, self.bias)
        return x


class Standard2d(Conv2d):
    """
    Implements weight standardization with learnable gain.
    Paper: https://arxiv.org/abs/2101.08692.

    Note that this layer must *always* be followed by some form of normalization.
    """

    scale: T.Final[float]

    def __init__(self, *args, gamma=1.0, eps=1e-6, gain=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain))
        self.scale = float(gamma * self.weight[0].numel() ** -0.5)
        self.eps = eps

    @TX.override
    def forward(self, x):
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)

        x = self._padding_forward(x, self.kernel_size, self.stride, self.dilation)
        x = self._conv_forward(x, weight, self.bias)

        return x


class Separable2d(NormActivationMixin, nn.Module):
    """
    Impements a depthwise-seperable 2d convolution. This is a combination of a depthwise convolution and a pointwise
    convolution. The depthwise convolution is a convolution with a kernel size of 1x1xKxK and a pointwise convolution
    is a convolution with a kernel size of 1x1x1x1. The depthwise convolution is applied to each input channel
    individually and the pointwise convolution is applied to the output of the depthwise convolution. This is
    equivalent to a convolution with a kernel size of 1x1xKxK.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 1,
        bias: bool = True,
        depthwise: T.Callable[..., nn.Module] = Conv2d,
        pointwise: T.Callable[..., nn.Module] = Conv2d,
        **kwargs,
    ):
        """
        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        expansion
            Expansion factor for the middle channels.
        bias
            Whether to add a bias to the *pointwise* convolution.
        depthwise
            Function to create the depthwise convolution.
        pointwise
            Function to create the pointwise convolution.
        **kwargs
            Additional arguments passed to the *depthwise* convolution.
        """

        super().__init__()

        mid_channels = int(in_channels * expansion)

        if mid_channels % in_channels != 0:
            raise ValueError(
                f"Number of input channels ({in_channels}) must be a divisor of the middle channels ({mid_channels})"
            )

        # Depthwise (KxKx1) convolution
        self.depthwise = depthwise(
            in_channels,
            mid_channels,
            bias=False,
            groups=in_channels,
            **kwargs,
        )

        fvcore.nn.weight_init.c2_msra_fill(module=self.depthwise)

        # Pointwise (1x1xD) convolution
        self.pointwise = pointwise(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            groups=1,
        )

        fvcore.nn.weight_init.c2_msra_fill(self.pointwise)

        # Store in_channels and out_channels for exporting
        self.in_channels = in_channels
        self.out_channels = out_channels

    @TX.override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
