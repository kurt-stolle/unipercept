"""Modules for working with sequentual layers."""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn


class SequentialList(nn.Sequential):
    """A nn.Sequential that takes a list of tensors as input and returns a list of tensors as output."""

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def forward(self, x: T.List[torch.Tensor]) -> T.List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x
