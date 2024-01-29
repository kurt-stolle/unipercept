"""
Implements various ``torchdata.datapipe`` pipelines.
"""

from __future__ import annotations

import typing as T

import typing_extensions as TX

from .cache import *
from .io_wrap import *
from .join import *
from .pattern import *
