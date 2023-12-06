"""
This module implements building blocks for building neural networks in PyTorch.
"""

from __future__ import annotations

from unicore.utils.module import lazy_module_factory

__all__ = ["backbones", "layers", "losses", "typings"]
__getattr__, __dir__ = lazy_module_factory(__name__, __all__)

del lazy_module_factory
