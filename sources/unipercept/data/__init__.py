"""This module implements dataset interfaces and dataloading."""

from __future__ import annotations

from . import collect, io, ops, points, types
from ._config import *
from ._helpers import *
from ._loader import *
from ._sampler import *


def __getattr__(name: str):
    import importlib

    if name == "sets":
        return importlib.import_module(f"{__name__}.sets")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
