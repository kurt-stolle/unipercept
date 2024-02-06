"""Implements a dataclass that represents a depth map, with support for transforms."""

from __future__ import annotations

import typing as T

import torch
from PIL import Image as pil_image
from torchvision.io import ImageReadMode
from typing_extensions import deprecated

from unipercept import file_io

if T.TYPE_CHECKING:
    import unipercept as up

__all__ = [
    "read_image",
    "ImageReadMode",
    "read_segmentation",
    "read_optical_flow",
    "read_depth_map",
]

MAX_CACHE: T.Final = 1000


@deprecated("Use Image.read instead.")
def read_image(path: str) -> up.data.tensors.Image:
    """Read an image from the disk."""
    from unipercept.data.tensors import Image

    return Image.read(path)


@deprecated("Use PanopticMap.read instead.")
def read_segmentation(
    path: str, info: up.data.sets.Metadata | None, /, **meta_kwds: T.Any
) -> up.data.tensors.PanopticMap:
    from unipercept.data.tensors import PanopticMap

    return PanopticMap.read(path, info, **meta_kwds)


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
