import functools
import math
import typing as T

import torch
from torch import nn

from unipercept.nn.activations import ActivationSpec, get_activation
from unipercept.nn.init import InitMixin, InitMode, InitSpec
from unipercept.nn.norms import NormSpec, get_default_bias, get_norm
from unipercept.types import Tensor

from .utils import NormActivationSupport

__all__ = [
    "Conv2d",
    "Linear2d",
    "ConvTranspose2d",
    "Standard2d",
    "Separable2d",
    "IntParam1d",
    "IntParam2d",
    "IntParam3d",
    "PaddingParam",
    "PaddingMode",
]

type IntParam1d = int | tuple[int] | T.Sequence[int]
type IntParam2d = int | tuple[int, int] | T.Sequence[int]
type IntParam3d = int | tuple[int, int, int] | T.Sequence[int]
type PaddingParam = T.Literal["valid", "same"] | int
type PaddingMode = T.Literal["zeros", "circular", "reflect"]


class Conv2d(NormActivationSupport, InitMixin, nn.Conv2d):
    r"""
    A 2D convolutional layer with optional normalization and activation.

    Notes
    -----
    Current benchmarks show that this is the fastest way to implement a convolutional
    layer with added normalization and activation, as it avoids the overhead of
    using a `Sequential` container. This may become redundant in the future as the
    PyTorch framework evolves.
    """

    norm: nn.Module | None
    activation: nn.Module | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntParam2d,
        *,
        bias: bool | None = None,
        norm: NormSpec | None = None,
        activation: ActivationSpec | None = None,
        **kwargs,
    ):
        norm_layer = get_norm(norm, out_channels)
        act_layer = get_activation(activation)
        bias = get_default_bias(bias, norm_layer)

        super().__init__(in_channels, out_channels, kernel_size, bias=bias, **kwargs)

        if norm_layer is not None:
            self.norm = norm_layer
        else:
            self.register_module("norm", None)
        if act_layer is not None:
            self.activation = act_layer
        else:
            self.register_module("activation", None)

        self.reset_parameters()

    @T.override
    def forward(self, input: Tensor) -> Tensor:
        result = self._conv_forward(input, self.weight, self.bias)
        if self.norm is not None:
            result = self.norm(result)
        if self.activation is not None:
            result = self.activation(result)
        return result


class Linear2d(Conv2d):
    r"""
    A convolutional layer with a kernel size of 1. This is equivalent to a linear layer,
    but with spatial dimensions.

    This is a convenience class that constraints the initialization to not have
    any parameters that would make it inconsistent with its intended purpose of being
    a linear mapping (kernel size, padding, etc.).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        bias: bool | None = None,
        norm: NormSpec | None = None,
        activation: ActivationSpec | None = None,
        init_mode: InitMode | None = None,
        init_spec: InitSpec | None = None,
    ):
        if out_channels is None:
            out_channels = in_channels
        super().__init__(
            in_channels,
            out_channels,
            1,
            padding=0,
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
            norm=norm,
            activation=activation,
            init_mode=init_mode,
            init_spec=init_spec,
        )


class ConvTranspose2d(nn.ConvTranspose2d):
    norm: nn.Module | None
    activation: nn.Module | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args,
        bias: bool | None = None,
        norm: NormSpec | None = None,
        activation: ActivationSpec | None = None,
        **kwargs,
    ):
        norm_layer = get_norm(norm, out_channels)
        act_layer = get_activation(activation)
        bias = get_default_bias(bias, norm_layer)

        super().__init__(in_channels, out_channels, *args, bias=bias, **kwargs)

        if norm_layer is not None:
            self.norm = norm_layer
        else:
            self.register_module("norm", None)
        if act_layer is not None:
            self.activation = act_layer
        else:
            self.register_module("activation", None)

        self.reset_parameters()

    @T.override
    def forward(self, input: Tensor, output_size: list[int] | None = None) -> Tensor:
        result = self._transpose_conv_forward(input, output_size)
        if self.norm is not None:
            result = self.norm(result)
        if self.activation is not None:
            result = self.activation(result)
        return result

    def _transpose_conv_forward(
        self, input: Tensor, output_size: list[int] | None
    ) -> Tensor:
        r"""
        Copied from `torch.nn.modules.conv.ConvTranspose2d` to allow for custom forwards.
        """
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]

        return nn.functional.conv_transpose2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )

    @T.override
    def reset_parameters(self) -> None:
        weight = self.weight.data
        feature = math.ceil(weight.size(2) / 2)
        channels = (2 * feature - 1 - feature % 2) / (2.0 * feature)
        for i in range(weight.size(2)):
            for j in range(weight.size(3)):
                weight[0, 0, i, j] = (1 - math.fabs(i / feature - channels)) * (
                    1 - math.fabs(j / feature - channels)
                )
        for channels in range(1, weight.size(0)):
            weight[channels, 0, :, :] = weight[0, 0, :, :]


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

    @T.override
    def forward(self, input: Tensor) -> Tensor:
        weight = nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)
        return self._conv_forward(input, weight, self.bias)


class Separable2d(NormActivationSupport, InitMixin, nn.Module):
    """
    Impements a depthwise-seperable 2d convolution. This is a combination of a depthwise convolution and a pointwise
    convolution. The depthwise convolution is a convolution with a kernel size of 1x1xKxK and a pointwise convolution
    is a convolution with a kernel size of 1x1x1x1. The depthwise convolution is applied to each input channel
    individually and the pointwise convolution is applied to the output of the depthwise convolution. This is
    equivalent to a convolution with a kernel size of 1x1xKxK.
    """

    in_channels: T.Final[int]
    hidden_channels: T.Final[int]
    out_channels: T.Final[int]
    reverse: T.Final[bool]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntParam2d,
        *,
        stride: IntParam2d = 1,
        padding: IntParam2d | str = "valid",
        padding_mode: str = "zeros",
        dilation: IntParam2d = 1,
        groups: int | None = None,
        expansion: int = 1,
        bias: bool | None = None,
        norm: NormSpec | None = None,
        activation: ActivationSpec | None = None,
        init_mode: InitMode | None = None,
        init_spec: InitSpec | None = None,
        reverse: bool = False,
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
        groups
            Number of groups for the *depthwise* convolution.
            If a float is provided, it is interpreted as a fraction
            of the input channels.
            If None (default) this is equal to the number of input channels.
        init
            Initialization method for the weights.
        **kwargs
            Additional arguments passed to the *depthwise* convolution.
        """

        hidden_channels = int(in_channels * expansion)
        if hidden_channels % in_channels != 0:
            raise ValueError(
                f"Number of input channels ({in_channels}) must be a divisor of the middle channels ({hidden_channels})"
            )
        if groups is None:
            groups = in_channels
        if isinstance(groups, float):
            assert groups < 1.0
            groups = int(groups * in_channels)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.reverse = reverse

        depthwise = functools.partial(
            Conv2d,
            in_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            bias=False,
            groups=groups,
            init_mode=init_mode,
            init_spec=init_spec,
        )
        pointwise = functools.partial(
            Linear2d,
            hidden_channels,
            out_channels,
            bias=bias,
            init_mode=init_mode,
            init_spec=init_spec,
        )

        super().__init__()

        if not self.reverse:
            self.depthwise = depthwise()
            self.pointwise = pointwise(
                norm=norm,
                activation=activation,
            )
        else:
            self.depthwise = pointwise()
            self.pointwise = depthwise(
                norm=norm,
                activation=activation,
            )
        self.reset_parameters()

    def _separable_conv(self, input: Tensor) -> Tensor:
        if self.reverse:
            result = self.pointwise(input)
            result = self.depthwise(result)
        else:
            result = self.depthwise(input)
            result = self.pointwise(result)
        return result

    @T.override
    def forward(self, input: Tensor) -> Tensor:
        return self._separable_conv(input)
