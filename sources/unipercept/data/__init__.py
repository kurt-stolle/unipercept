"""This module implements dataset interfaces and dataloading."""

from __future__ import annotations

from unicore.utils.module import lazy_module_factory

from ._config import *
from ._helpers import *
from ._loader import *
from ._sampler import *

__all__ = []
__getattr__, __dir__ = lazy_module_factory(__name__, ["collect", "io", "ops", "tensors", "types", "sets"])

del lazy_module_factory
