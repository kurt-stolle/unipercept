"""
Wrap and register torchvision's tensor types.
"""


from __future__ import annotations

import typing as T

import torch
import typing_extensions as TX
from torch.utils._pytree import tree_flatten
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask

from .registry import pixel_maps

__all__ = ["Image", "Mask", "BoundingBoxes", "BoundingBoxFormat"]

pixel_maps.register(Image)
pixel_maps.register(Mask)
