from __future__ import annotations

import enum

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict
from timm.layers import trunc_normal_
from torch import Tensor
from typing_extensions import override
from unicore.utils.tensorclass import Tensorclass

from unipercept.nn.layers import MapMLP
from unipercept.nn.layers.ops import dynamic_conv2d
from unipercept.nn.layers.weight import init_trunc_normal_

__all__ = ["DepthPrediction", "DepthHead"]


class DepthPrediction(Tensorclass):
    maps: Tensor
    means: Tensor


class DepthHead(nn.Module):
    """
    Generates a depth map from a kernel embeddding and feature space
    """

    max_depth: Tensor

    def __init__(
        self,
        feature_key: str,
        kernel_keys: list[str] | tuple[str, str],
        kernel_dims: list[int] | tuple[int, int],
        max_depth: float,
        num_heads=4,
        normal_dims: int = 16,
    ):
        super().__init__()

        if len(kernel_keys) != 2:
            raise ValueError("Depth head requires two kernels.")
        # elif len(kernel_dims) != 2:
        #     raise ValueError("Depth head requires two kernel dimensions.")

        self.feature_key = feature_key
        self.kernel_keys = kernel_keys

        self.register_buffer(
            "max_depth", torch.tensor(max_depth, dtype=torch.float32, requires_grad=False), persistent=False
        )

        self.attention = nn.MultiheadAttention(normal_dims, num_heads=num_heads, batch_first=True)

        self.proj_q = nn.Linear(kernel_dims[0], normal_dims)
        self.proj_k = nn.Linear(kernel_dims[1], normal_dims)
        self.proj_v = nn.Linear(kernel_dims[0], normal_dims)

        self.to_mean = nn.Linear(normal_dims, 1)
        self.to_range = nn.Linear(normal_dims, 1)

    def _forward_mean_range(self, k_depth: Tensor, k_mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns the mean and range for depth denormalization

        Parameters
        ----------
        f_depth : Tensor
            Depth feature space
        k_depth : Tensor
            Depth kernel used to compute the depth map in dynamic convolution with the depth feature space
        k_mask : Tensor
            Mask kernel used to compute the segmentation mask in dynamic convolution with the mask feature space

        Returns
        -------
        Mean (batch x 1), range (batch x 1)
        """
        # Compute attention
        q = self.proj_q(k_depth)
        k = self.proj_k(k_mask)
        v = self.proj_v(k_depth)
        a, w = self.attention(q, k, v)

        # Compute mean and range
        m = self.to_mean(a).sigmoid() * self.max_depth
        r = self.to_range(a).sigmoid().log1p() * self.max_depth

        # Ensure range is positive
        return m, r

    @override
    def forward(
        self, features: TensorDict, kernels: TensorDict, return_means: bool = True
    ) -> tuple[Tensor, Tensor | None]:
        # Retrieve features and kernels
        k_depth = kernels.get(self.kernel_keys[0])
        k_mask = kernels.get(self.kernel_keys[1])
        f_depth = features.get(self.feature_key)

        # Compute mean and range
        m, r = self._forward_mean_range(k_depth, k_mask)
        d = dynamic_conv2d(f_depth, k_depth)

        # Values are mapped to [-1, 1]
        d = d.sigmoid() * 2.0 - 1.0
        # d = d.tanh()

        # Denormalization
        d = d * r.unsqueeze(-1).expand_as(d)
        d = d + m.unsqueeze(-1).expand_as(d)

        # Ensure values are in range
        # d = d.clamp(min=0.0).clamp(max=self.max_depth)
        d = d.relu()

        return d, m if return_means else None
