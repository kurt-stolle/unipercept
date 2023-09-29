"""UniPercept: A Python package for the analysis of perceptual data and the training of deep neural networks in PyTorch."""
from __future__ import annotations

__version__ = "3.2.2"

from . import data, modeling, trainer
from ._patch_tensordict_pytree import *

if __name__ == "__main__":
    from ..unicli import __main__ as _
