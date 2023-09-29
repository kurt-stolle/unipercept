from __future__ import annotations

import torch
from torchvision.io import ImageReadMode
from torchvision.io import read_image as _read_image
from unicore import file_io
from unipercept.data.points import Image

__all__ = ["read_image", "ImageReadMode"]


@torch.inference_mode()
@file_io.with_local_path(force=True)
def read_image(path: str, *, mode=ImageReadMode.RGB) -> Image:
    """Read an image from the disk."""

    img = _read_image(path, mode=mode)
    return img.as_subclass(Image)
