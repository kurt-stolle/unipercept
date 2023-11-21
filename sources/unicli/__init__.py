"""
UniCLI: A CLI for UniPercept
"""

from __future__ import annotations

__version__ = "1.0.0"

from . import describe, echo, profile, train, backbones
from ._cmd import *
from ._config import *
from ._info import *

__all__ = ["command", "__version__"]

if __name__ == "__main__":
    from .__main__ import *
