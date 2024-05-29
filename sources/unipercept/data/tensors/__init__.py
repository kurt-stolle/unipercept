"""Extends ``torchvision.tv_tensors`` with new datapoint types."""

from __future__ import annotations

from . import helpers, registry
from ._depth import *
from ._flow import *
from ._panoptic import *
from ._torchvision import *
