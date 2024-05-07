"""
Implements a Squeeze-and-Excitation layers (i.e. channel attention)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import parse_shape, rearrange
from einops.layers.torch import EinMix
from torch import Tensor
from typing_extensions import override

from .utils import make_divisible
from .weight import init_msra_fill_, init_xavier_fill_

__all__ = ["SqueezeExcite2d", "CircularExcite2d", "SelfExcite2d"]


class SqueezeExcite2d(nn.Module):
    """
    Squeeze-and-Excitation module.
    """

    def __init__(
        self,
        channels: int,
        ratio: float = 1.0 / 16,
        divisor: int = 8,
        bias: bool = True,
    ):
        super().__init__()

        reduction = make_divisible(channels * ratio, divisor, round_limit=0.0)

        self.map1 = nn.Conv2d(channels, reduction, kernel_size=1, bias=bias)
        self.norm = nn.GroupNorm(1, reduction)
        self.act = nn.ReLU(inplace=True)
        self.map2 = nn.Conv2d(reduction, channels, kernel_size=1, bias=bias)
        self.gate = nn.Sigmoid()

        init_msra_fill_(self.map1)
        init_xavier_fill_(self.map2)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((-2, -1), keepdim=True)
        x_se = self.map1(x_se)
        x_se = self.act(self.norm(x_se))
        x_se = self.map2(x_se)
        return x * self.gate(x_se)


class CircularExcite2d(nn.Module):
    """
    Squeeze-and-Excitation module that uses circular padding.
    """

    def __init__(self, channels: int, gamma=2, beta=1):
        super().__init__()
        kernel_size = int(abs(math.log(channels, 2) + beta) / gamma)
        kernel_size = max(kernel_size if kernel_size % 2 else kernel_size + 1, 3)

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=False)
        self.gate = nn.Sigmoid()

    @override
    def forward(self, x):
        y = x.mean((-2, -1)).view(x.shape[0], 1, -1)
        y = nn.functional.pad(y, (self.padding, self.padding), mode="circular")
        y = self.conv(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)


class SelfExcite2d(nn.Module):
    """
    Squeeze-and-Excitation module that uses self-attention.
    """

    def __init__(self, channels: int, divisor: int = 8, heads: int = 1, bias=True):
        super().__init__()

        dims = channels // divisor

        if dims % heads != 0:
            msg = f"Query/Key dimension (channels/divisor = {channels}/{divisor} = {dims}) must be divisible by number of heads {heads}."
            raise ValueError(msg)

        self.n_heads = heads
        self.w_qk = EinMix(
            "b c_i h w -> b (h w) c_e",
            weight_shape="c_i c_e",
            bias_shape="c_e" if bias else None,
            c_i=channels,
            c_e=dims * 2,
        )
        self.w_v = EinMix(
            "b c_i h w -> b (h w) c_e",
            weight_shape="c_i c_e",
            bias_shape="c_e" if bias else None,
            c_i=channels,
            c_e=channels,
        )
        self.gamma = nn.Parameter(torch.zeros([1]))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.w_qk, self.w_v):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is None:
                continue
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.gamma)

    def merge_heads(self, v: Tensor) -> Tensor:
        return rearrange(v, "b hw (n_h c_h) -> (b n_h) hw c_h", n_h=self.n_heads)

    @override
    def forward(self, x: Tensor) -> Tensor:
        # Map input to query, key, and value
        Q, K = self.w_qk(x).chunk(2, dim=-1)
        V = self.w_v(x)

        # Split heads into batch dimension
        Q, K, V = map(self.merge_heads, (Q, K, V))

        # Transpose key for dot product
        K = K.transpose(-2, -1)

        # Compute attention weights
        energy = torch.bmm(Q, K)
        scale = Q.shape[-1] ** -0.5
        attention = nn.functional.softmax(energy * scale, dim=-1)

        # Compute output
        out = torch.bmm(attention, V)
        out = rearrange(out, "b (h w) c -> b c h w", **parse_shape(x, "b c h w"))
        out = x + out * self.gamma

        return out
