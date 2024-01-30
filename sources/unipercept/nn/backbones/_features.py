"""
Feature extraction utilities for PyTorch models, wraps the extended version provided by `timm`.
"""


from __future__ import annotations

import typing as T

import timm.models
import torch.nn as nn
import torchvision.models.feature_extraction

__all__ = [
    "register_notrace_module",
    "register_notrace_function",
    "create_feature_extractor",
]

# ------------ #
# Leaf modules #
# ------------ #

_leaf_modules = set()


def register_notrace_module(module: T.Type[nn.Module]):
    """
    Any module not under timm.models.layers should get this decorator if we don't want to trace through it.
    """
    _leaf_modules.add(module)
    return module


def is_notrace_module(module: T.Type[nn.Module]):
    return module in _leaf_modules or timm.models.is_notrace_module(module)


def get_notrace_modules():
    return list(_leaf_modules) + timm.models.get_notrace_modules()


# ------------------ #
# Autowrap functions #
# ------------------ #
_autowrap_functions = set()


def register_notrace_function(func: T.Callable):
    """
    Decorator for functions which ought not to be traced through
    """
    _autowrap_functions.add(func)
    return func


def is_notrace_function(func: T.Callable):
    return func in _autowrap_functions or timm.models.is_notrace_function(func)


def get_notrace_functions():
    return list(_autowrap_functions) + timm.models.get_notrace_functions()


def create_feature_extractor(
    model: nn.Module, return_nodes: T.Union[T.Dict[str, str], T.List[str]]
):
    return torchvision.models.feature_extraction.create_feature_extractor(
        model,
        return_nodes,
        tracer_kwargs={
            "leaf_modules": get_notrace_modules(),
            "autowrap_functions": get_notrace_functions(),
        },
    )
