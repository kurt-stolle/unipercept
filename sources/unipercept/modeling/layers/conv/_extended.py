from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from typing_extensions import override

from .utils import NormActivationMixin, PaddingMixin

__all__ = ["Conv2d"]


class Conv2d(NormActivationMixin, PaddingMixin, nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self._padding_forward(x, self.kernel_size, self.stride, self.dilation)
        x = self._conv_forward(x, self.weight, self.bias)
        return x
