"""
Wrap and register torchvision's tensor types.
"""


import typing as T

import torch
from torch.utils._pytree import tree_flatten
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask

from .registry import pixel_maps

__all__ = ["Image", "Mask", "BoundingBoxes", "BoundingBoxFormat"]

pixel_maps.register(Image)
pixel_maps.register(Mask)
