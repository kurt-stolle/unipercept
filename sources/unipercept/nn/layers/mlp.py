"""
Basic MLP and variants
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
import typing_extensions as TX

from unipercept.nn.layers.activation import ActivationSpec, get_activation
from unipercept.nn.layers.norm import NormSpec, get_default_bias, get_norm
from unipercept.nn.layers.utils import to_2tuple
from unipercept.nn.layers.weight import get_nonlinearity_name

__all__ = ["MLP"]

EPS = torch.finfo(torch.float32).eps


class MLP(nn.Module):
    """
    Multi-layer Perception layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | float = 1.0,
        *,
        dropout: float | T.Sequence[float] | T.Tuple[float, float] = 0.0,
        layers: int = 2,
        norm: (
            NormSpec
            | T.Sequence[NormSpec]
            | T.Tuple[NormSpec, NormSpec]
            | T.Tuple[NormSpec, NormSpec, NormSpec]
        ) = None,
        bias: T.Tuple[bool, bool] | bool | None = None,
        activation: (
            ActivationSpec
            | T.Sequence[ActivationSpec]
            | T.Tuple[ActivationSpec, ActivationSpec]
            | T.Tuple[ActivationSpec, ActivationSpec, ActivationSpec]
        ) = nn.GELU,
        init_gain: float | None = None,
    ):
        """
        Parameters
        ----------
        in_features
            Number of input channels.
        out_features
            Number of output channels.
        hidden_channels
            Number of hidden channels.
            If a float is provided, it is interpreted as a percentage of the input channels.
        layers
            The number of layers. Default: 2.
        dropout
            Dropout probability. Default: 0.0,
            If a tuple is provided, it is interpreted as (intra, post).
        norm
            Normalization layer to use.
            If `None`, no normalization is applied.
            If a tuple is provided, it is interpreted as (intra, post) or (antre, intra, post).
        bias
            Whether to add a bias term.
            If two values are provided, the final value is for the last layer.
            If `None`, a bias is added if no normalization is applied.
        activation
            Activation function to use. Default: `nn.GELU`.
            If a tuple is provided, it is interpreted as (intra, post) or (ante, intra, post).
            If `None`, no activation is applied.
        init_gain
            The gain value for the initialization of the weights. If `None`, no additional initialization is applied.
        """
        super().__init__()

        assert init_gain is None or init_gain >= 0.0, init_gain
        assert layers > 0

        # Handle hidden channels (if float, interpret as percentage of input channels)
        if isinstance(hidden_features, float):
            hidden_features = int(in_features * hidden_features)
        elif isinstance(hidden_features, int):
            pass
        else:
            msg = f"Invalid type for `hidden_channels`: {type(hidden_features)}"
            raise ValueError(msg)

        # Set attributes
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        # Handle variable parameters
        ante_norm, intra_norm, post_norm = _decompose_ante_intra_post(norm)
        ante_activation, intra_activation, post_activation = _decompose_ante_intra_post(
            activation
        )

        ante_bias, intra_bias, post_bias = _decompose_ante_intra_post(bias)
        if ante_bias is not None:
            msg = f"Ante bias is not supported. Got {bias=}"
            raise NotImplementedError(msg)
        ante_dropout, intra_dropout, post_dropout = _decompose_ante_intra_post(
            dropout, default=(None, 0.0, 0.0)
        )
        if ante_dropout is not None:
            msg = f"Ante dropout is not supported. Got {dropout=}"
            raise NotImplementedError(msg)

        # Construct layers
        layer_modules: list[MLPLayer] = []
        for n in range(layers):
            is_final = n >= (layers - 1)
            n_norm = post_norm if is_final else intra_norm
            n_activation = post_activation if is_final else intra_activation
            n_dropout = post_dropout if is_final else intra_dropout
            n_bias = post_bias if is_final else intra_bias
            n_init_gain = init_gain if is_final else None

            layer_modules.append(
                MLPLayer(
                    in_features if n == 0 else hidden_features,
                    out_features if is_final else hidden_features,
                    bias=n_bias,
                    norm=n_norm,
                    activation=n_activation,
                    dropout=n_dropout,
                    init_gain=n_init_gain,
                )
            )

        # Register modules
        if ante_norm is not None:
            self.input_norm = get_norm(ante_norm, in_features)
        else:
            self.register_module("input_norm", None)
        if ante_activation is not None:
            self.input_activation = get_activation(ante_activation)
        else:
            self.register_module("input_activation", None)
        assert len(layer_modules) == layers, (len(layer_modules), layers)
        self.layers = nn.Sequential(*layer_modules)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the MLP.
        """

        for layer in (self.input_activation, self.input_norm, *self.layers):
            try:
                layer.reset_parameters()
            except (AttributeError, NotImplementedError):
                pass

    @TX.override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.input_activation is not None:
            x = self.input_activation(x)
        return self.layers(x)


class MLPLayer(nn.Module):
    """
    See :class:`MLP` for more details.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool | None = None,
        norm: NormSpec = None,
        activation: ActivationSpec,
        dropout: float,
        init_gain: float | None,
    ):
        super().__init__()

        self.init_gain = init_gain

        norm = get_norm(norm, out_features)
        bias = get_default_bias(bias, norm)
        linear = nn.Linear(in_features, out_features, bias=bias)
        if norm is None or isinstance(norm, nn.Identity):
            self.linear = nn.utils.parametrizations.weight_norm(linear, name="weight")
            self.register_module("norm", None)
        else:
            self.linear = linear
            self.norm = norm

        activation = get_activation(activation)
        if activation is None:
            self.register_module("activation", None)
        else:
            self.activation = activation

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for layer in (self.norm, self.activation, self.dropout):
            try:
                layer.reset_parameters()
            except (AttributeError, NotImplementedError):
                pass

        # When init_gain is specified, the user requests an initialization where the
        # output of the layer is roughly equal to the input scaled by the gain.
        if self.activation is None or self.init_gain is not None:
            nn.init.xavier_uniform_(self.linear.weight, gain=self.init_gain or 1.0)
        else:
            nonlin_name, nonlin_slope = get_nonlinearity_name(self.activation)
            nn.init.kaiming_uniform_(
                self.linear.weight,
                nonlinearity=nonlin_name,
                a=nonlin_slope,
                mode="fan_out",
            )
        # Zero out bias if present
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    @TX.override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.dropout(x)


_Ti = T.TypeVar("_Ti", covariant=True)
_Ta = T.TypeVar("_Ta", covariant=True)
_Tp = T.TypeVar("_Tp", covariant=True)


def _decompose_ante_intra_post(
    arg: T.Sequence[_Ti | _Ti | _Tp] | tuple[_Ti, _Ta, _Tp] | tuple[_Ti, _Tp] | _Ti,
    *,
    default: tuple[_Ti, _Ta, _Tp] = (None, None, None),
) -> T.Tuple[_Ta | _Ti, _Ti, _Tp | _Ti]:
    """
    Quick macro for handling the decomposition of antre- intera- and pre- arguments.
    """
    if isinstance(arg, T.Sequence):
        *ante_intra, post = arg
        if len(ante_intra) == 1:
            intra = ante_intra[0]
            ante = default[0]
        else:
            ante, intra = ante_intra
    else:
        ante, intra, post = default

    return ante, intra, post  # type: ignore
