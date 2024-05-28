"""
Basic MLP and variants
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from typing_extensions import override

from unipercept.nn.layers.activation import ActivationSpec, get_activation
from unipercept.nn.layers.norm import NormSpec, get_norm
from unipercept.nn.layers.utils import to_2tuple

__all__ = ["MLP"]

EPS = torch.finfo(torch.float32).eps


class MLP(nn.Sequential):
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
            Number of hidden channels. If a float is provided, it is interpreted as a percentage of the input channels.
        layers
            The number of layers. Default: 2.
        dropout
            Dropout probability. Default: 0.0,
            If a tuple is provided, it is interpreted as (intra, post).
        norm
            Normalization layer to use. If `None`, no normalization is applied.
            If a tuple is provided, it is interpreted as (intra, post) or (antre, intra, post).
        bias
            Whether to add a bias term. If two values are provided, the final value is for the last layer. If `None`, a bias is added if no normalization is applied.
        activation
            Activation function to use. Default: `nn.GELU`.
            If a tuple is provided, it is interpreted as (intra, post) or (ante, intra, post).
            If `None`, no activation is applied.
        init_gain
            The gain value for the initialization of the weights. If `None`, no additional initialization is applied.
        """
        super().__init__()

        self.layers = layers
        self.init_gain = init_gain

        # Handle hidden channels (if float, interpret as percentage of input channels)
        if isinstance(hidden_features, float):
            hidden_features = int(in_features * hidden_features)
        elif isinstance(hidden_features, int):
            pass
        else:
            msg = f"Invalid type for `hidden_channels`: {type(hidden_features)}"
            raise ValueError(msg)

        # Handle variable parameters
        ante_norm, intra_norm, post_norm = _decompose_ante_intra_post(
            norm, default=None
        )
        if ante_norm is not None:
            self.add_module("norm_0", get_norm(ante_norm, in_features))

        ante_activation, intra_activation, post_activation = _decompose_ante_intra_post(
            activation, default=None
        )
        if ante_activation is not None:
            self.add_module("act_0", get_activation(ante_activation))

        ante_bias, intra_bias, post_bias = _decompose_ante_intra_post(
            bias, default=None
        )
        if ante_bias is not None:
            msg = f"Ante bias is not supported. Got {bias=}"
            raise NotImplementedError(msg)
        if intra_bias is None:
            intra_bias = intra_norm is None
        if post_bias is None:
            post_bias = post_norm is None

        ante_dropout, intra_dropout, post_dropout = _decompose_ante_intra_post(
            dropout, default=None
        )
        if ante_dropout is not None:
            msg = f"Ante dropout is not supported. Got {dropout=}"
            raise NotImplementedError(msg)

        # Construct layers
        for n in range(layers):
            is_final = n >= (layers - 1)
            n_norm = post_norm if is_final else intra_norm
            n_activation = post_activation if is_final else intra_activation
            n_dropout = post_dropout if is_final else intra_dropout
            n_bias = post_bias if is_final else intra_bias

            # Fully connected
            fc = nn.Linear(
                in_features if n == 0 else hidden_features,
                hidden_features if not is_final else out_features,
                bias=n_bias,
            )
            if is_final and init_gain is not None:
                nn.init.normal_(fc.weight, mean=0.0, std=init_gain)
                if fc.bias is not None:
                    nn.init.zeros_(fc.bias)
            self.add_module(f"map_{n+1}", fc)

            if n_activation is not None:
                self.add_module(f"act_{n+1}", get_activation(n_activation))
            if n_dropout is not None:
                self.add_module(f"drop_{n+1}", nn.Dropout(n_dropout, inplace=False))
            if n_norm is not None:
                self.add_module(
                    f"norm_{n+1}",
                    get_norm(n_norm, out_features if is_final else hidden_features),
                )


_T = T.TypeVar("_T")


def _decompose_ante_intra_post(
    arg: T.Sequence[_T] | _T, *, default: _T
) -> T.Tuple[_T, _T, _T]:
    f"""
    Quick macro for handling the decomposition of antre- intera- and pre- arguments.
    """
    if isinstance(arg, T.Sequence):
        *ante_intra, post = arg
        if len(ante_intra) == 1:
            intra = ante_intra[0]
            ante = default
        else:
            ante, intra = ante_intra
    else:
        intra = arg
        ante = post = default

    return ante, intra, post
