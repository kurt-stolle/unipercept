"""
This module implements building blocks for building neural networks in PyTorch.
"""

from __future__ import annotations

import typing as T

import typing_extensions as TX

from unipercept.utils.module import lazy_module_factory

__all__ = []
__getattr__, __dir__ = lazy_module_factory(
    __name__, ["backbones", "layers", "losses", "typings", "integrations"]
)

del lazy_module_factory
