from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from ._extended import Conv2d

__all__ = ["Standard2d"]


class Standard2d(Conv2d):
    """
    Implements weight standardization with learnable gain.
    Paper: https://arxiv.org/abs/2101.08692.

    Note that this layer must *always* be followed by some form of normalization.
    """

    scale: T.Final[float]

    def __init__(self, *args, gamma=1.0, eps=1e-6, gain=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain))
        self.scale = float(gamma * self.weight[0].numel() ** -0.5)
        self.eps = eps

    @override
    def forward(self, x):
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_channels, -1),
            None,
            None,
            weight=(self.gain * self.scale).view(-1),
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)

        x = self._padding_forward(x, self.kernel_size, self.stride, self.dilation)
        x = self._conv_forward(x, weight, self.bias)

        return x
