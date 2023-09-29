import torch
from typing_extensions import override


class WeightedLoss(torch.nn.Module):
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
