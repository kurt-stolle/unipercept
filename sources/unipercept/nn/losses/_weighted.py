"""
Implements a weighted loss.


This module is included mostly for backward compatability. It is recommended to
implement a weighted loss using the functional API or directly in the model definition.

The weighted loss module does not work well with TorchScript, therefore it is not
recommended to use it in production. Issues are mainly related to the fact that
the loss function uses a variable number of arguments, which is not supported by
TorchScript. We use some hacks to get around this issue, but it is not guaranteed
to work in all cases.
"""

from __future__ import annotations

import typing as T

import torch
from typing_extensions import override, deprecated

__all__ = ["WeightedLoss"]


class _WeightedLoss(torch.nn.Module):
    """
    Simple wrapper around a loss function to weight it, useful for
    configuration files that need specify a loss function
    for multiple tasks.
    """

    def __init__(self, loss: torch.nn.Module, weight: float):
        super().__init__()
        self.loss = loss
        self.weight = weight

    @override
    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs) * self.weight


_L = T.TypeVar("L", bound=torch.nn.Module)


@deprecated("Module based weighting does not support compilation and scripting.")
def WeightedLoss(loss: _L, weight: float) -> _L:
    """
    Functional variant for instantiating a weighted loss with proper typing support.
    """
    return T.cast(_L, _WeightedLoss(loss, weight))
