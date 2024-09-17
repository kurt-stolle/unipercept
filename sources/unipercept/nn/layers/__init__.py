"""Layers that can generalically be used in a neural network."""

from __future__ import annotations

import typing as T
import warnings

import typing_extensions as TX

from . import (
    conv,
    deform_attn,
    deform_conv,
    dropout,
    dynamic_conv,
    layer_scale,
    linear,
    position,
    projection,
    squeeze_excite,
)
from ._coord import *
from ._interpolate import *
from ._sequential import *


def __getattr__(name: str):
    """
    Hotfix legacy imports
    """
    import importlib

    DEPRECATE_MAP = {
        "mlp": "unipercept.nn.layers.linear._mlp",
        "activation": "unipercept.nn.activations",
        "norm": "unipercept.nn.norms",
        "weight": "unipercept.nn.init",
    }

    if (module := DEPRECATE_MAP.get(name)) is not None:
        msg = f"Module {__name__}.{name} has been moved to {module}"
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

        return importlib.import_module(module)

    msg = f"Module {__name__} has no attribute {name}"
    raise AttributeError(msg)
