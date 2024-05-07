"""Layers that can generalically be used in a neural network."""

from __future__ import annotations

import typing as T
import warnings

import typing_extensions as TX

from . import activation, conv, norm, projection, squeeze_excite, utils, weight
from ._coord import *
from ._interpolate import *
from ._mlp import *
from ._sequential import *


def __getattr__(name: str):
    """
    Hotfix legacy imports
    """

    if name == "SqueezeExcite2d":
        msg = "The `SqueezeExcite2d` layer has been moved to `unipercept.nn.layers.squeeze_excite`."
        warnings.warn(msg, stacklevel=0)

        return squeeze_excite.SqueezeExcite2d
