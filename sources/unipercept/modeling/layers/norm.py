from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override


class GroupNorm32(nn.GroupNorm):
    """Wrap the group norm used in Detectron2, where a fixed group size of 32 is always hardcoded."""

    def __init__(self, num_channels: int, *, num_groups: int = 32, **kwargs) -> None:
        super().__init__(num_groups, num_channels, **kwargs)


def layer_norm_chw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    x = weight[:, None, None] * x + bias[:, None, None]
    return x


class LayerNormCHW(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    @override
    def forward(self, x):
        return layer_norm_chw(x.float(), self.weight, self.bias, self.eps).type_as(x)


def global_response_norm(
    x: torch.Tensor,
    spatial_dim: tuple[int, int],
    channel_dim: int,
    bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_g = x.float().norm(p=2, dim=spatial_dim, keepdim=True)
    x_n = x_g / (x_g.mean(dim=channel_dim, keepdim=True) + eps)
    return x + torch.addcmul(bias.float(), weight.float(), x.float() * x_n).type_as(x)


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization based on ``ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders``.

    Paper: https://arxiv.org/abs/2301.00808
    """

    def __init__(self, channels, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        self.weight = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    @override
    def forward(self, x):
        return global_response_norm(
            x,
            self.spatial_dim,
            self.channel_dim,
            self.bias.view(self.wb_shape),
            self.weight.view(self.wb_shape),
            self.eps,
        )


class GlobalResponseNorm2d(GlobalResponseNorm):
    """Gobal response norm for CHW tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, channels_last=False, **kwargs)
