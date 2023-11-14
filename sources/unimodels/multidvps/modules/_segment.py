"""
Generates masks using dynamic convolutions.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import override
from unicore.utils.tensorclass import Tensorclass

from unipercept.nn.layers.ops import dynamic_conv2d

if T.TYPE_CHECKING:
    from torch import Tensor

__all__ = ["GeneratedInstances", "MaskHead"]


class GeneratedInstances(Tensorclass):
    logits: Tensor
    fused_kernels: TensorDict
    categories: T.Optional[Tensor]
    scores: T.Optional[Tensor]


class MaskHead(nn.Module):
    key: T.Final[str]

    def __init__(self, key: str):
        super().__init__()

        self.key = key

    @override
    def forward(
        self,
        features: TensorDict,
        kernels: TensorDict,
    ) -> Tensor:
        logits = self.dynamic_conv(features, kernels)

        return logits

    def dynamic_conv(self, features: TensorDict, kernels: TensorDict):
        """
        Perform dynamic convolution on the features.
        """
        k = kernels.get(self.key)
        f = features.get(self.key)
        return dynamic_conv2d(f, k)
