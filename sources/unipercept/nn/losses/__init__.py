"""
This module hosts various losses for perception tasks.
"""

from __future__ import annotations

import typing as T

import typing_extensions as TX

from . import functional, mixins
from ._contrastive import *
from ._depth import *
from ._focal import *
from ._guided import *
from ._image import *
from ._panoptic import *
