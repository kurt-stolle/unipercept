"""
This module implements building blocks for building neural networks in PyTorch.
"""

from unipercept.utils.module import lazy_module_factory

from ._base import *
from ._builtin import *

__all__ = []
__getattr__, __dir__ = lazy_module_factory(
    __name__, ["gradclip", "multitask", "precisebn", "stop"]
)

del lazy_module_factory
