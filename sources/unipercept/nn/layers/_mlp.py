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
from unipercept.nn.layers.utils import to_ntuple

__all__ = ["MapMLP"]

EPS = torch.finfo(torch.float32).eps


class MapMLP(nn.Sequential):
    """
    Straightforward MLP that maps the multipurpose kernel to a task-specific kernel.

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
    eps
        Epsilon value for normalization layers. Defaults to the machine epsilom for float32.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | float = 1.0,
        *,
        dropout: float | T.Iterable[float] = 0.0,
        layers: int = 3,
        norm: NormSpec | None = None,
        bias=True,
        activation: ActivationSpec = nn.GELU,
        init_gain: float = 1.0,
    ):
        super().__init__()

        if isinstance(hidden_channels, float):
            hidden_channels = int(in_channels * hidden_channels)
        elif isinstance(hidden_channels, int):
            pass
        else:
            raise ValueError(
                f"Invalid type for `hidden_channels`: {type(hidden_channels)}"
            )

        dropout = to_ntuple(layers)(dropout)

        for n, d in enumerate(dropout):
            is_final = n == layers - 1
            fc = nn.Linear(
                in_channels if n == 0 else hidden_channels,
                hidden_channels if n < layers - 1 else out_channels,
                bias=bias if norm is None else False,
            )
            if is_final:
                nn.init.orthogonal_(fc.weight, init_gain)
            else:
                nn.init.kaiming_normal_(fc.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(fc.bias)
            self.add_module(f"fc{n}", fc)
            if not is_final:
                self.add_module(f"act{n}", get_activation(activation))
            self.add_module(f"drop{n}", nn.Dropout(d, inplace=True))
            if norm is not None and not is_final:
                self.add_module(f"norm{n}", get_norm(norm, hidden_channels))
