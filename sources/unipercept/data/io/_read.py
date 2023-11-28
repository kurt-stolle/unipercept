"""Implements a dataclass that represents a depth map, with support for transforms."""

from __future__ import annotations

import typing as T
import functools
import torch
from torchvision.io import ImageReadMode
from PIL import Image as pil_image
from torchvision.transforms.v2.functional import to_image, to_dtype
from typing_extensions import deprecated
from unicore import file_io

if T.TYPE_CHECKING:
    import unipercept as up

__all__ = ["read_image", "ImageReadMode", "read_segmentation", "read_optical_flow", "read_depth_map"]

MAX_CACHE: T.Final = 1000


# @functools.lru_cache(maxsize=MAX_CACHE)
@file_io.with_local_path(force=True)
def read_image(path: str, *, mode=ImageReadMode.RGB) -> up.data.tensors.Image:
    """Read an image from the disk."""

    from unipercept.data.tensors import Image

    with pil_image.open(path) as img_pil:
        img_pil = img_pil.convert("RGB")
        img = to_image(img_pil)
        img = to_dtype(img, torch.float32, scale=True)

    assert img.shape[0] == 3, f"Expected image to have 3 channels, got {img.shape[0]}!"

    return img


# @functools.lru_cache(maxsize=MAX_CACHE)
@deprecated("Use PanopticMap.read instead.")
def read_segmentation(
    path: str, info: up.data.sets.Metadata | None, /, **meta_kwds: T.Any
) -> up.data.tensors.PanopticMap:
    from unipercept.data.tensors import PanopticMap

    return PanopticMap.read(path, info, **meta_kwds)


# @functools.lru_cache(maxsize=MAX_CACHE)
@file_io.with_local_path(force=True)
def read_optical_flow(path: str) -> torch.Tensor:
    from flowops import Flow

    from unipercept.data.tensors import OpticalFlow

    flow = torch.from_numpy(Flow.read(path).as_2hw())
    return OpticalFlow(flow.to(dtype=torch.float32))


@deprecated("Use DepthMap.read instead.")
def read_depth_map(path: str, *args, **kwargs) -> up.data.tensors.DepthMap:
    from unipercept.data.tensors import DepthMap

    return DepthMap.read(path, *args, **kwargs)
