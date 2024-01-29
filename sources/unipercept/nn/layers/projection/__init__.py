"""
This package contains some classes from the Pytorch3D package. The combination
of Python and PyTorch versions required for the other packages in this project
did now allow the installation of the PyTorch3D from their official channels.

See: https://github.com/facebookresearch/pytorch3d
"""

from __future__ import annotations

import typing as T

import typing_extensions as TX

from . import render, transform
from .module import *
