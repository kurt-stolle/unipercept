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

__all__ = ["MapMLP"]

EPS = torch.finfo(torch.float32).eps


class MapMLP(nn.Sequential):
    """
    Straightforward MLP.

    Taken from VisionTransformer

    Parameters
    ----------
    in_channels
        Number of input channels.
    out_channels
        Number of output channels.
    hidden_channels
        Number of hidden channels. If a float is provided, it is interpreted as a percentage of the input channels.
    dropout
        Dropout probability.
    norm
        Normalization layer to use. If `None`, no normalization is applied.
    bias
        Whether to add a bias term.
    activation
        Activation function to use. If `None`, no activation is applied. Default: `nn.GELU`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | float = 1.0,
        *,
        dropout: float | T.Iterable[float] = 0.0,
        layers: int = 2,
        norm: NormSpec | None = None,
        bias: T.Tuple[bool, bool] | bool | None = None,
        activation: ActivationSpec = nn.GELU,
        init_gain: float | None = None,
    ):
        """
        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        hidden_channels
            Number of hidden channels. If a float is provided, it is interpreted as a percentage of the input channels.
        layers
            The number of layers. Default: 2.
        dropout
            Dropout probability. If two values are provided, the final value is for the last layer.
        norm
            Normalization layer to use. If `None`, no normalization is applied.
        bias
            Whether to add a bias term. If two values are provided, the final value is for the last layer. If `None`, a bias is added if no normalization is applied.
        activation
            Activation function to use. If `None`, no activation is applied. Default: `nn.GELU`.
        init_gain
            The gain value for the initialization of the weights. If `None`, no additional initialization is applied.
        """
        super().__init__()

        if layers == 1:
            msg = "At least two layers are required."
            raise ValueError(msg)

        if isinstance(hidden_channels, float):
            hidden_channels = int(in_channels * hidden_channels)
        elif isinstance(hidden_channels, int):
            pass
        else:
            msg = f"Invalid type for `hidden_channels`: {type(hidden_channels)}"
            raise ValueError(msg)

        if isinstance(bias, T.Iterable):
            bias, final_bias = bias
        else:
            final_bias = bias
        if bias is None:
            bias = norm is None
        if final_bias is None:
            final_bias = True

        if isinstance(dropout, T.Iterable):
            dropout, final_dropout = dropout
        else:
            final_dropout = 0.0

        for n in range(layers):
            is_final = n >= (layers - 1)

            # Fully connected layer
            fc = nn.Linear(
                in_channels if n == 0 else hidden_channels,
                hidden_channels if not is_final else out_channels,
                bias=bias if not is_final else final_bias,
            )
            if is_final and init_gain is not None:
                nn.init.orthogonal_(fc.weight, gain=init_gain)
                if fc.bias is not None:
                    nn.init.zeros_(fc.bias)
            self.add_module(f"fc{n}", fc)

            # Activation
            if not is_final and activation is not None:
                self.add_module(f"act{n}", get_activation(activation))

            # Dropout
            layer_dropout = final_dropout if is_final else dropout
            if layer_dropout > 0.0:
                self.add_module(
                    f"drop{n}", nn.Dropout(layer_dropout, inplace=False)
                )  # (not is_final)))

            # Normalization
            if norm is not None and not is_final:
                self.add_module(f"norm{n}", get_norm(norm, hidden_channels))
