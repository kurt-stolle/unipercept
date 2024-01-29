"""
A collection of reusable utilities for UniPercept
"""

from __future__ import annotations

import typing as T

import typing_extensions as TX

from unipercept.utils.module import lazy_module_factory

__all__ = [
    "abbreviate",
    "box",
    "camera",
    "dicttools",
    "function",
    "inspect",
    "logutils",
    "mask",
    "seed",
    "state",
    "tensor",
    "time",
]  # type: ignore
__getattr__, __dir__ = lazy_module_factory(__name__, __all__)

del lazy_module_factory
