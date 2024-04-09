"""
Implements a Squeeze-and-Excitation layer.
"""

from __future__ import annotations

import torch.nn as nn
from typing_extensions import override

from .utils import make_divisible

__all__ = ["SqueezeExcite2d"]


class SqueezeExcite2d(nn.Module):
    """
    Squeeze-and-Excitation module.
    """

    def __init__(self, channels, ratio=1.0 / 16, divisor=8, bias=True):
        super().__init__()

        reduction = make_divisible(channels * ratio, divisor, round_limit=0.0)

        self.map1 = nn.Conv2d(channels, reduction, kernel_size=1, bias=bias)
        self.norm = nn.GroupNorm(1, reduction)
        self.act = nn.ReLU(inplace=True)
        self.map2 = nn.Conv2d(reduction, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

    @override
    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.map1(x_se)
        x_se = self.act(self.norm(x_se))
        x_se = self.map2(x_se)
        return x * self.gate(x_se)
