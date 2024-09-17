"""
Implements a Squeeze-and-Excitation layers (i.e. channel attention)
"""

import math
from typing import override

import torch
from einops import parse_shape, rearrange
from einops.layers.torch import EinMix
from torch import Tensor, nn

from unipercept.nn.activations import ActivationSpec, InplaceReLU6
from unipercept.nn.init import InitMode
from unipercept.nn.layers.conv import Linear2d

from .._args import make_divisible

__all__ = ["SqueezeExcite2d", "CircularExcite2d", "SelfExcite2d"]


class SqueezeExcite2d(nn.Module):
    """
    Squeeze-and-Excitation module.
    """

    def __init__(
        self,
        channels: int,
        ratio: float = 1 / 2,
        divisor: int = 8,
        activation: ActivationSpec = InplaceReLU6,
        scale: float | None = 0.0,
    ):
        super().__init__()

        reduction = make_divisible(channels * ratio, divisor, round_limit=0.0)

        self.squeeze = Linear2d(
            channels,
            reduction,
            bias=False,
            activation=activation,
            init_mode=InitMode.C2_NORMAL,
        )
        self.excite = Linear2d(
            reduction,
            channels,
            bias=False,
            activation=nn.Sigmoid,
        )

        if scale is not None:
            self.scale = nn.Parameter(torch.tensor([scale]))
        else:
            self.register_parameter("scale", None)

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = nn.functional.adaptive_avg_pool2d(input, (1, 1))
        weight = self.excite(self.squeeze(weight))
        output = input * weight

        if self.scale is not None:
            output = output * self.scale
        return output


class CircularExcite2d(nn.Module):
    """
    Squeeze-and-Excitation module that uses circular padding.
    """

    def __init__(self, channels: int, gamma=2, beta=1, scale: float | None = 0.0):
        super().__init__()
        kernel_size = int(abs(math.log2(channels) + beta) / gamma)
        kernel_size = max(kernel_size if kernel_size % 2 else kernel_size + 1, 3)

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=0, bias=False)
        self.gate = nn.Sigmoid()

        if scale is not None:
            self.scale = nn.Parameter(torch.tensor([scale]))
        else:
            self.register_parameter("scale", None)

    @override
    def forward(self, input):
        weight = input.mean((-2, -1)).view(input.shape[0], 1, -1)
        weight = nn.functional.pad(
            weight, (self.padding, self.padding), mode="circular"
        )
        weight = self.conv(weight)
        weight = self.gate(weight).view(input.shape[0], -1, 1, 1)
        output = input * weight.expand_as(input)

        if self.scale is not None:
            output = output * self.scale
        return output


class SelfExcite2d(nn.Module):
    """
    Squeeze-and-Excitation module that uses self-attention.
    """

    def __init__(
        self,
        channels: int,
        divisor: int = 8,
        heads: int = 1,
        bias=True,
        scale: float | None = 0.0,
    ):
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

        if scale is not None:
            self.scale = nn.Parameter(torch.tensor([scale]))
        else:
            self.register_parameter("scale", None)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in (self.w_qk, self.w_v):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is None:
                continue
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.scale)

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

        if self.scale is not None:
            out = out * self.scale
        return out
