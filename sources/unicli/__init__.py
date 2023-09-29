"""
UniCLI: A CLU for UniPercept
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import check, describe, echo, infer, train
from ._cmd import *
from ._config import *
from ._info import *

__all__ = ["command", "__version__"]
