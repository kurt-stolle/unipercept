"""
Defines the types of data that can be used in the data pipeline, consisting of mostly TypedDicts and Enums.
"""

from __future__ import annotations

import typing as T

import typing_extensions as TX

from . import sanity
from ._coco import *
from ._info import *
from ._manifest import *
