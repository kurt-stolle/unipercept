"""
Wrap and register torchvision's tensor types.
"""


from __future__ import annotations


from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat, Image, Mask

from .registry import pixel_maps

__all__ = ["Image", "Mask", "BoundingBoxes", "BoundingBoxFormat"]

pixel_maps.register(Image)
pixel_maps.register(Mask)
