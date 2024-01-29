from __future__ import annotations

import typing as T

import typing_extensions as TX
from torchvision.tv_tensors import Mask

from .registry import pixel_maps

__all__ = ["OpticalFlow"]


@pixel_maps.register
class OpticalFlow(Mask):
    pass
