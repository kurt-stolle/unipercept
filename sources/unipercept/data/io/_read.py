"""Implements a dataclass that represents a depth map, with support for transforms."""

from __future__ import annotations

import typing as T

import torch
from torchvision.io import ImageReadMode
from torchvision.io import read_image as _read_image
from typing_extensions import deprecated
from unicore import file_io

if T.TYPE_CHECKING:
    import unipercept as up

__all__ = ["read_image", "ImageReadMode", "read_segmentation", "read_optical_flow", "read_depth_map"]


@file_io.with_local_path(force=True)
def read_image(path: str, *, mode=ImageReadMode.RGB) -> up.data.tensors.Image:
    """Read an image from the disk."""

    from unipercept.data.tensors import Image

    img = _read_image(path, mode=mode)
    return img.as_subclass(Image)


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
