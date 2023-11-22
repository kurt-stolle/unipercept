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
    feature_key: T.Final[str]
    geometry_key: T.Final[str]

    def __init__(
        self,
        feature_key: str,
        geometry_key: str,
        max_depth: float,
        min_depth: float = 1.0,
        num_heads: int = 4,
        dropout=0.0,
    ):
        super().__init__()

        self.feature_key = feature_key
        self.geometry_key = geometry_key

        self.max_depth = max_depth
        self.min_depth = min_depth

    def _forward_mean_range(self, k_geom: Tensor) -> tuple[Tensor, Tensor]:
        """
        Returns the mean and range for depth denormalization
        """
        # mr_abs = self.min_depth + k_geom * (self.max_depth - self.min_depth)
        _, mr_abs = self._convert_to_absolute_depth(k_geom)

        m, r = mr_abs.chunk(2, dim=-1)

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
        k_depth = kernels.get(self.feature_key)
        k_geom = kernels.get(self.geometry_key)
        f_depth = features.get(self.feature_key)

        # Compute mean and range
        m, r = self._forward_mean_range(k_geom)
        d = dynamic_conv2d(f_depth, k_depth)

        # Values are mapped to [-1, 1]
        d = d.sigmoid() * 2.0 - 1.0
        # d = d.tanh()

        # Denormalization
        d = d * r.unsqueeze(-1).expand_as(d)
        d = d + m.unsqueeze(-1).expand_as(d)

        # Ensure values are in range
        # d = d.clamp(min=0.0).clamp(max=self.max_depth)
        # d = d.clamp(self.min_depth, self.max_depth)
        if self.training:
            d = d.relu()
        else:
            d = d.clamp(self.min_depth, self.max_depth)

        return d, m if return_means else None
