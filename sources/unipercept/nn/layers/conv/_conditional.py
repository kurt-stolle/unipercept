"""
Conditionally Parameterized Convolution.

Paper: https://arxiv.org/abs/1904.04971
"""

from __future__ import annotations

import math
from functools import partial
from typing import override

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from unipercept.nn.layers.conv.utils import NormActivationMixin
from unipercept.utils.function import to_2tuple

__all__ = ["CondConv2d"]


def get_condconv_initializer(initializer, num_experts, expert_shape):
    def condconv_initializer(weight):
        """CondConv initializer function."""
        num_params = np.prod(expert_shape)
        if (
            len(weight.shape) != 2
            or weight.shape[0] != num_experts
            or weight.shape[1] != num_params
        ):
            raise (
                ValueError(
                    "CondConv variables must have shape [num_experts, num_params]"
                )
            )
        for i in range(num_experts):
            initializer(weight[i].view(expert_shape))

    return condconv_initializer


class CondConv2d(NormActivationMixin, nn.Module):
    r"""
    Conditional convolution layer.
    """

    __constants__ = ["in_channels", "out_channels"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding: int | tuple[int, int] = 0,
        dilation=1,
        groups=1,
        bias=False,
        num_experts=4,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.padding = to_2tuple(padding)
        self.dilation = to_2tuple(dilation)
        self.groups = groups
        self.num_experts = num_experts

        self.weight_shape = (
            self.out_channels,
            self.in_channels // self.groups,
        ) + self.kernel_size
        weight_num_param = 1
        for wd in self.weight_shape:
            weight_num_param *= wd
        self.weight = torch.nn.Parameter(
            torch.torch.Tensor(self.num_experts, weight_num_param)
        )

        if bias:
            self.bias_shape = (self.out_channels,)
            self.bias = torch.nn.Parameter(
                torch.torch.Tensor(self.num_experts, self.out_channels)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        init_weight = get_condconv_initializer(
            partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
            self.num_experts,
            self.weight_shape,
        )
        init_weight(self.weight)
        if self.bias is not None:
            fan_in = np.prod(self.weight_shape[1:])
            bound = 1 / math.sqrt(fan_in)
            init_bias = get_condconv_initializer(
                partial(nn.init.uniform_, a=-bound, b=bound),
                self.num_experts,
                self.bias_shape,
            )
            init_bias(self.bias)

    @override
    def forward(self, x, routing_weights):
        B, C, H, W = x.shape
        weight = torch.matmul(routing_weights, self.weight)
        new_weight_shape = (
            B * self.out_channels,
            self.in_channels // self.groups,
        ) + self.kernel_size
        weight = weight.view(new_weight_shape)
        bias = None
        if self.bias is not None:
            bias = torch.matmul(routing_weights, self.bias)
            bias = bias.view(B * self.out_channels)
        # move batch elements with channels so each batch element can be efficiently convolved with separate kernel
        # reshape instead of view to work with channels_last input
        x = x.reshape(1, B * C, H, W)
        x = F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * B,
        )
        x = x.permute([1, 0, 2, 3]).view(B, self.out_channels, x.shape[-2], x.shape[-1])
        return x
