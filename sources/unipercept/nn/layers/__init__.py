"""Layers that can generalically be used in a neural network."""

from __future__ import annotations

import typing as T
import warnings

import typing_extensions as TX

from . import (
    activation,
    conv,
    deform_attn,
    deform_conv,
    flash_attn,
    norm,
    position,
    projection,
    squeeze_excite,
    utils,
    weight,
    mlp,
)
from ._coord import *
from ._interpolate import *
from ._sequential import *


def __getattr__(name: str):
    """
    Hotfix legacy imports
    """

    if name == "SqueezeExcite2d":
        msg = "The `SqueezeExcite2d` layer has been moved to `unipercept.nn.layers.squeeze_excite`."
        warnings.warn(msg, stacklevel=1)

        return squeeze_excite.SqueezeExcite2d
    if name == "MapMLP":
        msg = "The `MapMLP` layer has been moved to `unipercept.nn.layers.mlp.MLP`."
        warnings.warn(msg, stacklevel=1)

        return mlp.MLP

    msg = f"Module {__name__} has no attribute {name}"
    raise AttributeError(msg)
