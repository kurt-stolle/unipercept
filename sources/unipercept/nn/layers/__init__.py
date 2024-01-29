"""Layers that can generalically be used in a neural network."""
from __future__ import annotations

import typing as T

import typing_extensions as TX

from unipercept.utils.module import lazy_module_factory

from ._attention import *
from ._coord import *
from ._interpolate import *
from ._mlp import *
from ._sequential import *
from ._squeeze_excite import *

__all__ = []
__getattr__, __dir__ = lazy_module_factory(
    __name__, ["conv", "merge", "norm", "projection", "tracking", "utils"]
)

del lazy_module_factory
