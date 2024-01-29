"""Extends ``torchvision.tv_tensors`` with new datapoint types."""

from __future__ import annotations

import typing as T

import typing_extensions as TX

from . import helpers, registry
from ._depth import *
from ._flow import *
from ._panoptic import *
from ._torchvision import *
