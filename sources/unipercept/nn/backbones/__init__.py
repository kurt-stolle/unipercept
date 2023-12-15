"""
Implements various backbones for feature extraction.
"""

from __future__ import annotations

from . import fpn, fpn_legacy, timm, torchvision
from ._base import *
from ._features import *
from ._normalize import *
