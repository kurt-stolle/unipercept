"""
Normalization layers. These are mostly wrappers around PyTorch's norm layers, enabling straightforward configuration 
through the config system.
"""

from __future__ import annotations

import functools
import typing as T

import torch
import torch.nn as nn
from typing_extensions import override

from unipercept.utils.inspect import locate_object

NormFactory: T.TypeAlias = T.Callable[[int], nn.Module]
NormSpec: T.TypeAlias = str | NormFactory | nn.Module | None


def get_norm(spec: NormSpec, num_channels: int, **kwargs) -> nn.Module:
    """
    Resolve a norm module from a string, a factory function or a module instance. When a module instance is provided,
    the ``num_channels`` argument is ignored and the object is returned as-is.

    Parameters
    ----------
    norm
        A string, a factory function or an instance of a norm module.
    num_channels
        Number of channels to normalize.
    **kwargs
        Additional keyword arguments to pass to the norm factory.

    """

    if spec is None:
        spec = nn.Identity()
    elif isinstance(spec, str):
        spec = locate_object(spec)

    # If already a module instance, return that instance directly (num_channels is ignored)
    if isinstance(spec, nn.Module):
        return spec
    elif callable(spec):
        return spec(num_channels, **kwargs)
    else:
        raise ValueError(f"Cannot resolve value as a norm module: {spec}")


def GroupNorm32(num_channels: int, **kwargs) -> nn.GroupNorm:
    """
    GroupNorm with the number of groups equal to 32, like in Detectron2.

    Notes
    -----
    The amount of channels' value (32) originates from the optimal value found in the GroupNorm paper. This is
    only the case for the specific families tested, and may be suboptimal for other specific cases.
    """
    return nn.GroupNorm(32, num_channels=num_channels, **kwargs)


def GroupNormCG(num_channels: int, **kwargs) -> nn.GroupNorm:
    """
    GroupNorm with the number of groups equal to the number of channels.
    """
    return nn.GroupNorm(num_channels, num_channels=num_channels, **kwargs)


def LayerNormCHW(num_channels: int, **kwargs) -> nn.GroupNorm:
    """
    LayerNorm with the number of groups equal to the number of channels.
    """
    return nn.GroupNorm(num_groups=1, num_channels=num_channels, **kwargs)


def GroupNormFactory(*, num_groups: int, **kwargs) -> T.Callable[[int], nn.GroupNorm]:
    """
    GroupNorm with the number of groups equal to the number of channels.
    """
    return functools.partial(nn.GroupNorm, num_groups, **kwargs)


# @torch.jit.script
# def layer_norm_chw(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
#     u = x.mean(1, keepdim=True)
#     s = (x - u).pow(2).mean(1, keepdim=True)
#     x = (x - u) / torch.sqrt(s + eps)
#     x = weight[:, None, None] * x + bias[:, None, None]
#     return x


# class LayerNormCHW(nn.Module):
#     def __init__(self, normalized_shape, eps=1e-6):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.normalized_shape = (normalized_shape,)

#     @override
#         return layer_norm_chw(x.float(), self.weight, self.bias, self.eps).type_as(x)


@torch.jit.script
def global_response_norm(
    x: torch.Tensor,
    spatial_dim: T.Tuple[int, int],
    channel_dim: int,
    bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_i = x.type_as(weight)
    x_g = x_i.norm(p=2, dim=spatial_dim, keepdim=True)
    x_n = x_g / (x_g.mean(dim=channel_dim, keepdim=True) + eps)
    y = x_i + torch.addcmul(bias, weight, x_i * x_n)

    return y.type_as(x)


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
