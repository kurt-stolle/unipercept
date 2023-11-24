"""
Wrap and register torchvision's tensor types.
"""


from torchvision.tv_tensors import Image, Mask, BoundingBoxes, BoundingBoxFormat
import typing as T
from .registry import pixel_maps
import torch

from torch.utils._pytree import tree_flatten

__all__ = ["Image", "Mask", "BoundingBoxes", "BoundingBoxFormat"]

pixel_maps.register(Image)
pixel_maps.register(Mask)
