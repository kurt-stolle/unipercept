"""Extends ``torchvision.tv_tensors`` with new datapoint types."""

from . import helpers, registry
from ._camera import *
from ._depth import *
from ._flow import *
from ._mask import *
from ._panoptic import *
from ._torchvision import *
