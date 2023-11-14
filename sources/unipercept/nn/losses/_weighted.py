import typing as T

import torch
from typing_extensions import override

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


L = T.TypeVar("L", bound=torch.nn.Module)


def WeightedLoss(loss: L, weight: float) -> L:
    """
    Functional variant for instantiating a weighted loss with proper typing support.
    """
    return T.cast(L, _WeightedLoss(loss, weight))
