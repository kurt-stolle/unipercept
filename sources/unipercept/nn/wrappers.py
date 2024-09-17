"""
Implements wrapper functions for modifying models on the fly.
"""

from __future__ import annotations

import typing as T

import regex as re
from torch import nn

from unipercept.log import logger


def freeze_parameters[_M: nn.Module](
    module: _M, pattern: str | re.Pattern[str] | None = None, requires_grad=False
) -> _M:
    """
    Freeze the parameters of a module.

    Parameters
    ----------
    module
        The module to freeze.
    pattern
        A regular expression pattern to match the parameter names that should be frozen.
        If None, all parameters are frozen.
    requires_grad
        Whether to freeze the parameters or not.
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    for name, param in module.named_parameters():
        if pattern is not None and not pattern.match(name):
            continue
        logger.debug(f"Freezing parameter: {name}")
        param.requires_grad = requires_grad

    return module


def freeze_batchnorm[_M: nn.Module](module: _M, **kwargs) -> _M:
    """
    Freeze the batch normalization layers of a module.

    Parameters
    ----------
    module
        The module to freeze.
    """
    from unipercept.nn.norms import FrozenBatchNorm2d

    return T.cast(_M, FrozenBatchNorm2d.convert_from(module, **kwargs))


def batchnorm_to_groupnorm[_M: nn.Module](module: _M, num_groups: int = 32) -> _M:
    """
    Convert the batch normalization layers of a module to group normalization.

    Parameters
    ----------
    module
        The module to convert.
    """
    for name, submod in module.named_modules():
        if isinstance(submod, nn.BatchNorm2d):
            num_channels = submod.num_features
            gn = nn.GroupNorm(
                min(num_groups, num_channels), num_channels, eps=submod.eps, affine=True
            )
            gn.weight.data.copy_(submod.weight.data)
            gn.bias.data.copy_(submod.bias.data)

            setattr(module, name, gn)
    return module
