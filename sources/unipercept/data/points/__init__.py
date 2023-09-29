"""Extends ``torchvision.datapoints`` with new datapoint types."""

from __future__ import annotations

from torchvision import disable_beta_transforms_warning as __disable_warning

__disable_warning()

from . import registry
from ._camera import *
from ._depth import *
from ._flow import *
from ._image import *
from ._mask import *
from ._model import *
from ._panoptic import *
