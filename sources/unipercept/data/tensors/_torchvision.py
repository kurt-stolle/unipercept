"""
Wrap and register torchvision's tensor types.
"""


from __future__ import annotations

import typing as T

import PIL.Image as pil_image
import torch
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat
from torchvision.tv_tensors import Image as ImageBase
from torchvision.tv_tensors import Mask

from unipercept.utils.typings import Pathable

from .registry import pixel_maps

__all__ = ["Image", "Mask", "BoundingBoxes", "BoundingBoxFormat"]


class Image(ImageBase):
    """
    Extension of the ``Image`` tensor in Torchvision.
    """

    @classmethod
    def read(cls, path: Pathable) -> T.Self:
        """Reads an image from the given path."""
        from torchvision.transforms.v2.functional import to_dtype, to_image

        from unipercept.file_io import get_local_path

        path = get_local_path(str(path))

        with pil_image.open(path) as img_pil:
            img_pil = img_pil.convert("RGB")
            img = to_image(img_pil)
            img = to_dtype(img, torch.float32, scale=True)

        assert (
            img.shape[0] == 3
        ), f"Expected image to have 3 channels, got {img.shape[0]}!"

        return img.as_subclass(cls)


pixel_maps.register(ImageBase)
pixel_maps.register(Image)
pixel_maps.register(Mask)
