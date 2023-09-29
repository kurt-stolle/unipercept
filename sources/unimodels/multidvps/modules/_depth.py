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
from unipercept.modeling.layers.ops import dynamic_conv2d
from unipercept.modeling.layers.weight import init_trunc_normal_

__all__ = ["DepthPrediction", "DepthHead"]


class DepthPrediction(Tensorclass):
    maps: Tensor
    means: Tensor


class IDN(enum.StrEnum):
    GAUSSIAN = "gaussian"
    SKEWED = "skewed"
    MIXTURE = "mixture"


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
        idn=IDN.GAUSSIAN,
    ):
        super().__init__()

        if len(kernel_keys) != 2:
            raise ValueError("Depth head requires two kernels.")
        elif len(kernel_dims) != 2:
            raise ValueError("Depth head requires two kernel dimensions.")

        self.idn = IDN(idn)

        self.feature_key = feature_key
        self.kernel_keys = kernel_keys

        self.register_buffer(
            "max_depth", torch.tensor(max_depth, dtype=torch.float32, requires_grad=False), persistent=False
        )

        params_bilinear = nn.Bilinear(kernel_dims[0], kernel_dims[1], 2, bias=True)
        nn.utils.parametrizations.spectral_norm(params_bilinear, n_power_iterations=2, eps=1e-6, name="weight")

        self.mlp_normal = nn.Sequential(
            params_bilinear,
            nn.Sigmoid(),
        )
        self.mlp_normal.apply(init_trunc_normal_)

    def _init_weights(
        self, m: nn.Module, norm_mean: float = 0.0, norm_std: float = 0.02, norm_rng: float = 1.0, bias: float = 0.0
    ) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if hasattr(m, "weight"):
                trunc_normal_(m.weight, mean=norm_mean, std=norm_std, a=-norm_rng, b=norm_rng)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, bias)

    def _split_embedding(self, kernels: TensorDict) -> tuple[Tensor, Tensor, Tensor]:
        """
        Split the mean and range values from the depth kernel.
        """

        e = kernels.get(self.kernel_keys[0])
        m, r = (
            self.mlp_normal(kernels.get(self.kernel_keys[0]), kernels.get(self.kernel_keys[1])) * self.max_depth
        ).chunk(2, dim=-1)
        r = r / 2.0

        return e, m, r

    @override
    def forward(
        self,
        features: TensorDict,
        kernels: TensorDict,
    ) -> tuple[Tensor, Tensor]:
        e, m, r = self._split_embedding(kernels)
        d = dynamic_conv2d(features.get(self.feature_key), e)

        # Values are mapped to [-1, 1]
        # d = d.sigmoid() * 2.0 - 1.0
        d = d.tanh()

        # Denormalization
        d = d * r.unsqueeze(-1).expand_as(d)
        d = d + m.unsqueeze(-1).expand_as(d)

        # Ensure values are in range
        # d = d.clamp(min=0.0).clamp(max=self.max_depth)
        d = d.relu()

        return d, m
