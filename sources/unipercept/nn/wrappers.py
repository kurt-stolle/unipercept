"""
Implements wrapper functions for modifying models on the fly.
"""

from __future__ import annotations

import torch.nn as nn
import torch

import regex as re

import typing as T

_M = T.TypeVar("_M", bound=nn.Module)


def freeze_parameters(module: _M, pattern: str | re.Pattern[str] | None = None) -> _M:
    """
    Freeze the parameters of a module.

    Parameters
    ----------
    module
        The module to freeze.
    pattern
        A regular expression pattern to match the parameter names that should be frozen.
        If None, all parameters are frozen.
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    for name, param in module.named_parameters():
        if pattern is not None and not pattern.match(name):
            continue
        param.requires_grad = False

    return module


def freeze_batchnorm(module: _M, **kwargs) -> _M:
    """
    Freeze the batch normalization layers of a module.

    Parameters
    ----------
    module
        The module to freeze.
    """
    from unipercept.nn.layers.norm import FrozenBatchNorm2d

    return T.cast(_M, FrozenBatchNorm2d.convert_from(module, **kwargs))
