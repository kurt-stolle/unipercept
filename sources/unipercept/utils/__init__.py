"""
A collection of reusable utilities for UniPercept
"""

from __future__ import annotations

from unipercept.utils.module import lazy_module_factory

__all__ = [
    "abbreviate",
    "box",
    "geometry",
    "check",
    "dicttools",
    "function",
    "inspect",
    "logutils",
    "mask",
    "seed",
    "state",
    "tensor",
    "time",
    "memory",
    "cli",
]  # type: ignore
__getattr__, __dir__ = lazy_module_factory(__name__, __all__)

del lazy_module_factory
