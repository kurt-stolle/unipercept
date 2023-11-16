from __future__ import annotations

import typing as T
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from typing_extensions import override
from unicore.utils.tensorclass import Tensorclass

from unipercept.nn.layers.ops import dynamic_conv2d

__all__ = ["DepthPrediction", "DepthHead"]


class DepthPrediction(Tensorclass):
    maps: Tensor
    means: Tensor


class DepthHead(nn.Module):
    """
    Generates a depth map from a kernel embeddding and feature space
    """

    max_depth: T.Final[float]
    min_depth: T.Final[float]

    def __init__(
        self,
        feature_key: str,
        kernel_keys: list[str] | tuple[str, str],
        kernel_dims: list[int] | tuple[int, int],
        max_depth: float,
        min_depth: float = 1.0,
        num_heads: int=4,
        normal_dims: int = 16,
        dropout=0.0
    ):
        super().__init__()

        if len(kernel_keys) != 2:
            raise ValueError("Depth head requires two kernels.")
        elif len(kernel_dims) != 2:
            raise ValueError("Depth head requires two kernel dimensions.")

        self.feature_key = feature_key
        self.kernel_keys = kernel_keys

        self.max_depth = max_depth
        self.min_depth = min_depth

        self.project = nn.Linear(kernel_dims[0], normal_dims)
        self.attention = nn.MultiheadAttention(normal_dims, kdim=kernel_dims[0], vdim=kernel_dims[1], dropout=dropout, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(normal_dims)
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
        n = self.project(k_depth)
        a, _ = self.attention(n, k_depth, k_mask, needs_weights=False)
        n = self.norm(n + a)


        # Compute mean and range
        m = self.to_mean(a).sigmoid() * self.max_depth
        r = self.to_range(a).sigmoid().log1p() * self.max_depth

        return m, r

    def _convert_to_absolute_depth(self, disparity: torch.Tensor) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert network output ([0,1]) into depth prediction ([0,max_depth])
        """
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disparity
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def _convert_to_normalized_disparity(self, depth: torch.Tensor) -> T.Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert depth prediction ([0,max_depth]) into network output ([0,1])
        """
        min_disp = 1 / self.max_depth
        max_disp = 1 / self.min_depth
        scaled_disp = 1 / depth
        disparity = (scaled_disp - min_disp) / (max_disp - min_disp)
        return scaled_disp, disparity

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
        d = d.clamp(self.min_depth, self.max_depth)

        return d, m if return_means else None
