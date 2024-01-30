from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import parse_shape, rearrange
from einops.layers.torch import EinMix
from torch import Tensor
from typing_extensions import override

__all__ = ["SelfAttention2d", "CircularChannelAttention2d"]


class SelfAttention2d(nn.Module):
    """
    Self-attention layer based on `Self-Attention Generative Adversarial Networks`.
    Paper: https://arxiv.org/abs/1805.08318.
    """

    def __init__(self, d_in: int, d_qk: int | None = None, n_heads: int = 1, bias=True):
        super().__init__()

        if d_qk is None:
            d_qk = d_in // 8

        if d_qk % n_heads != 0:
            raise ValueError(
                f"Query/Key dimension {d_qk} must be divisible by number of heads {n_heads}."
            )

        self.n_heads = n_heads
        self.w_qk = EinMix(
            "b c_i h w -> b (h w) c_e",
            weight_shape="c_i c_e",
            bias_shape="c_e" if bias else None,
            c_i=d_in,
            c_e=d_qk * 2,
        )
        self.w_v = EinMix(
            "b c_i h w -> b (h w) c_e",
            weight_shape="c_i c_e",
            bias_shape="c_e" if bias else None,
            c_i=d_in,
            c_e=d_in,
        )

        self.gamma = nn.Parameter(torch.zeros([1]))

    def _reset_parameters(self):
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
        attention = F.softmax(energy * scale, dim=-1)

        # Compute output
        # out = einsum(attention, V, "bh hw_q hw_kv, bh hw_kv c_h -> bh hw c_h")
        # out = rearrange(out, "(b n_h) hw c_h -> b hw (n_h c_h)", n_h=self.n_heads)
        out = torch.bmm(attention, V)
        out = rearrange(out, "b (h w) c -> b c h w", **parse_shape(x, "b c h w"))
        out = x + out * self.gamma

        return out


class CircularChannelAttention2d(nn.Module):
    """
    Multiplies each channel with a learned attention weighting matrix.
    Circular padding is applied in order to not treat the channel dimension as spatial.

    # TODO: Add support for multiple attention heads.
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
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)
        y = F.pad(y, (self.padding, self.padding), mode="circular")
        y = self.conv(y)
        y = self.gate(y).view(x.shape[0], -1, 1, 1)
        return x * y.expand_as(x)
