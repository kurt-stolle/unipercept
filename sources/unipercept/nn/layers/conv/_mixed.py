"""
Implementation of mixed convolutional layers.

Paper: https://arxiv.org/abs/1907.09595
"""


from __future__ import annotations

import torch
from torch import nn as nn
from typing_extensions import override

from ._extended import Conv2d


def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


class MixedConv2d(nn.ModuleDict):
    """Mixed group convolutional layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding="",
        dilation=1,
        depthwise=False,
        **kwargs,
    ):
        super().__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)
        for idx, (k, in_ch, out_ch) in enumerate(
            zip(kernel_size, in_splits, out_splits)
        ):
            conv_groups = in_ch if depthwise else 1
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                Conv2d(
                    in_ch,
                    out_ch,
                    k,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=conv_groups,
                    **kwargs,
                ),
            )
        self.splits = in_splits

    @override
    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x_split[i]) for i, c in enumerate(self.values())]
        x = torch.cat(x_out, 1)
        return x
