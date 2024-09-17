"""Modules for working with sequentual layers."""

from __future__ import annotations

import torch
from torch import nn


class SequentialList(nn.Sequential):
    """A nn.Sequential that takes a list of tensors as input and returns a list of tensors as output."""

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        for module in self:
            x = module(x)
        return x
