from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from typing_extensions import Self, override

from . import utils
from ._extended import Conv2d
from ._standard import Standard2d

__all__ = ["Separable2d"]


class Separable2d(nn.Module):
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
        *,
        expansion: int = 1,
        bias: bool = True,
        depthwise: Callable[..., nn.Module] = Conv2d,
        pointwise: Callable[..., nn.Module] = Conv2d,
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

        # Store in_channels and out_channels for exporting
        self.in_channels = in_channels
        self.out_channels = out_channels

    @classmethod
    def with_norm(cls, *args, norm: None | Callable[[int], nn.Module] = None, **kwargs) -> Self:
        return cls.with_norm_activation(norm=norm, activation=None)

    @classmethod
    def with_activation(cls, *args, activation: None | Callable[..., nn.Module] = None, **kwargs) -> Self:
        return cls.with_norm_activation(norm=None, activation=activation)

    @classmethod
    def with_norm_activation(
        cls,
        *args,
        norm: None | Callable[[int], nn.Module] = None,
        activation: None | Callable[..., nn.Module] = None,
        **kwargs,
    ) -> Self:
        if norm is not None:
            depthwise = kwargs.pop("depthwise", Standard2d)
            kwargs["depthwise"] = lambda *dw_args, **dw_kwargs: utils._init_sequential_norm(
                depthwise, *dw_args, **dw_kwargs, norm=norm
            )
        if activation is not None:
            pointwise = kwargs.pop("pointwise", Conv2d)
            kwargs["pointwise"] = lambda *pw_args, **pw_kwargs: utils._init_sequential_activation(
                pointwise, *pw_args, **pw_kwargs, activation=activation
            )

        return cls(
            *args,
            **kwargs,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
