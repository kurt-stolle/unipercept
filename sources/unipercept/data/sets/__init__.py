"""This module contains the dataset modules."""
from __future__ import annotations

from ._base import *
from ._helpers import *
from ._meta import *
from .cityscapes import DVPS as CityscapesVPS
from .cityscapes import Base as Cityscapes
from .pascal import VOC as PascalVOC


def register():
    import warnings

    # deprecated
    warnings.warn(
        f"Registration not required when using `get_{{info,dataset}}` from `{__name__}`.",
        DeprecationWarning,
        stacklevel=1,
    )
