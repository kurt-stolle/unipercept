"""
Implements the kernel mapper module, which takes maps the multipurpose kernel $k^\star$ to all the different
specific kernels using a simple dictionary of heads, represented as a `nn.ModuleDict`.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from typing_extensions import override

from unipercept.nn.layers.activation import ActivationSpec, get_activation
from unipercept.nn.layers.norm import NormSpec, get_norm

__all__ = ["MapMLP", "EmbedMLP"]

EPS = torch.finfo(torch.float32).eps


class MapMLP(nn.Module):
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

    eps: T.Final[float]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int | float = 1.0,
        *,
        dropout=0.0,
        norm: NormSpec = nn.LayerNorm,
        bias=True,
        activation: ActivationSpec = nn.GELU,
        eps: float = EPS,
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

        self.eps = eps
        self.fc1 = nn.Linear(
            in_channels, hidden_channels, bias=bias if norm is None else False
        )
        self.act = get_activation(activation)
        self.drop1 = nn.Dropout(dropout, inplace=True)
        self.norm = get_norm(norm, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels, bias=bias)
        self.drop2 = nn.Dropout(dropout, inplace=True)

    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop1(y)
        y = self.norm(y)
        y = self.fc2(y)
        y = self.drop2(y)

        return y

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_mlp(x)


class EmbedMLP(MapMLP):
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self._forward_mlp(x), p=2, dim=1, eps=self.eps)
