"""
Normalization layers. These are mostly wrappers around PyTorch's norm layers, enabling straightforward configuration
through the config system.
"""

from __future__ import annotations

import functools
import typing as T
from typing import override

import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d, SyncBatchNorm
from torchvision.ops.misc import FrozenBatchNorm2d as _FrozenBatchNorm2d

from unipercept.types import Size, Tensor
from unipercept.utils.inspect import locate_object

#############
# LayerNorm #
#############

LayerNorm = nn.LayerNorm
RMSNorm = nn.RMSNorm

_LAYERNORM_DEFAULT_EPS = 1e-6


def layer_norm(
    input: Tensor,
    shapes: Size,
    weight: Tensor,
    bias: Tensor,
    *,
    eps: float = _LAYERNORM_DEFAULT_EPS,
) -> Tensor:
    return nn.functional.layer_norm(input, shapes, weight, bias, eps=eps)


def rms_norm(
    input: Tensor,
    shapes: Size,
    weight: Tensor,
    *,
    eps=_LAYERNORM_DEFAULT_EPS,
) -> Tensor:
    return nn.functional.rms_norm(input, shapes, weight, eps=eps)


#################
# Norm Registry #
#################

type NormFactory = T.Callable[[int], nn.Module]
type NormSpec = str | NormFactory | type[nn.Module] | nn.Module | None


def get_default_bias(
    value: bool | None, norm: nn.Module | None, *, default: bool = True
) -> bool:
    """
    Check whether a norm module makes the bias of a module (e.g. Conv2d or Linear) redundant.

    A bias parameter is only needed of the norm module has been explicitly set to
    not include a bias term.
    """
    # If the value is a bool, return that value
    if isinstance(value, bool):
        return value
    if norm is None:
        return default

    assert isinstance(norm, nn.Module), f"{norm=} ({type(norm)}) is not a module"

    if hasattr(norm, "default_bias"):
        return norm.default_bias
    if hasattr(norm, "affine"):
        return not norm.affine
    if hasattr(norm, "bias"):
        return not norm.bias
    if hasattr(norm, "elementwise_affine"):
        return not norm.elementwise_affine

    return default


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
    if callable(spec):
        return spec(num_channels, **kwargs)
    raise ValueError(f"Cannot resolve value as a norm module: {spec}")


###############
# Group Norms #
###############


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
# def layer_norm_chw(x: Tensor, weight: Tensor, bias: Tensor, eps: float) -> Tensor:
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


#########################
# Global Respoonse Norm #
#########################


@torch.jit.script
def global_response_norm(
    x: Tensor,
    spatial_dim: tuple[int, int],
    channel_dim: int,
    bias: Tensor,
    weight: Tensor,
    eps: float,
) -> Tensor:
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


class FrozenBatchNorm2d(_FrozenBatchNorm2d):
    @classmethod
    def convert_from[_M: nn.Module](cls, module: _M, recursive=True) -> _M | T.Self:
        """
        Converts BatchNorm2d or SyncBatchNorm modules.

        See Also
        --------
        - Based on `SyncBatchNorm <https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py>`_.
        """
        res = module
        if isinstance(module, (BatchNorm2d, SyncBatchNorm)):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        elif recursive:
            for name, child in module.named_children():
                new_child = cls.convert_from(child, recursive=True)
                if new_child is not child:
                    res.add_module(name, new_child)
        else:
            msg = f"Cannot convert {module.__class__.__name__} to {cls.__name__}."
            raise ValueError(msg)

        del module
        return res
