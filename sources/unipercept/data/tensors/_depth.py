from __future__ import annotations

import typing as T
from enum import StrEnum, auto

import safetensors.torch as safetensors
import torch
from torch.types import Device
from torchvision.tv_tensors import Mask

from unipercept import file_io
from unipercept.data.tensors.helpers import get_kwd, read_pixels
from unipercept.data.tensors.registry import pixel_maps

__all__ = ["DepthMap", "DepthFormat"]

DEFAULT_DEPTH_DTYPE: T.Final = torch.float32


class DepthFormat(StrEnum):
    DEPTH_INT16 = auto()
    DISPARITY_INT16 = auto()
    TORCH = auto()
    SAFETENSORS = auto()


@pixel_maps.register
class DepthMap(Mask):
    @classmethod
    def default_like(cls, other: torch.Tensor) -> T.Self:
        """Returns a default instance of this class with the same shape as the given tensor."""
        return cls(torch.full_like(other, fill_value=0, dtype=torch.float32))

    @classmethod
    def default(cls, shape: torch.Size, device: Device = "cpu") -> T.Self:
        """Returns a default instance of this class with the given shape."""
        return cls(torch.zeros(shape, device=device, dtype=torch.float32))  # type: ignore

    @classmethod
    @torch.no_grad()
    def read(
        cls, path: str, dtype: torch.dtype = DEFAULT_DEPTH_DTYPE, **meta_kwds: T.Any
    ) -> T.Self:
        path = file_io.get_local_path(path)
        # Switch by depth format
        format = get_kwd(meta_kwds, "format", DepthFormat | str)
        match DepthFormat(format):  # type: ignore
            case DepthFormat.DEPTH_INT16:
                m = read_pixels(path, color=False).to(torch.float64)
                m /= float(2**8)
            case DepthFormat.DISPARITY_INT16:
                m = cls.read_from_disparity(path, **meta_kwds)
            case DepthFormat.SAFETENSORS:
                m = safetensors.load_file(path)["data"]
            case DepthFormat.TORCH:
                m = torch.load(path, map_location="cpu").squeeze_(0).squeeze_(0)
            case _:
                raise NotImplementedError(f"Unsupported depth format: {format}")

        # TODO: Add angular FOV compensation via metadata
        m = m.to(dtype=dtype)
        m[m == torch.inf] = 0.0
        m[m == torch.nan] = 0.0
        m.squeeze_()
        if m.ndim > 2:
            raise ValueError(f"Depth map has {m.ndim} dimensions, expected 2")

        return cls(m)

    @classmethod
    def read_from_disparity(
        cls,
        path: str,
        camera_baseline: float,
        camera_fx: float,
    ) -> T.Self:
        # Get machine epsilon for the given dtype, used to check for invalid values
        eps = torch.finfo(torch.float32).eps

        # Read disparity map
        disp = read_pixels(path, False)
        assert disp.dtype == torch.int32, disp.dtype

        # Convert disparity from 16-bit to float
        disp = disp.to(dtype=torch.float32, copy=False)
        disp -= 1
        disp[disp >= eps] /= 256.0

        # Infer depth using camera parameters and disparity
        valid_mask = disp >= eps

        depth = torch.zeros_like(disp)
        depth[valid_mask] = (camera_baseline * camera_fx) / disp[valid_mask]

        # Set invalid depth values to 0
        depth[depth == torch.inf] = 0
        depth[depth == torch.nan] = 0

        return cls(depth)


# def transform_depth_map(
#     depth_map: torch.Tensor,
#     transform: T.Transform,
#     augmentations: T.AugmentationList,
#     clip=True,
# ):
#     map_tf = transform.apply_segmentation(depth_map)

#     # Scale depth values to account for the new field of view caused by
#     # resize transformations
#     for aug in augmentations.augs:
#         if not isinstance(aug, T.ResizeTransform):
#             continue

#         # Tranform properties are defined in a way that makes the
#         # typechecker unable to access the attribute, hence why
#         # there values have their type ignored
#         scale_w = aug.w / aug.new_w  # type: ignore
#         scale_h = aug.h / aug.new_h  # type: ignore
#         scale = (scale_w + scale_h) / 2

#         # Clip depth values to minimum and maximum of the original values
#         if clip:
#             map_tf = torch.clip(map_tf * scale, map_tf.min(), map_tf.max())
#         else:
#             map_tf = map_tf * scale

#     if isinstance(map_tf, torch.ndarray):
#         map_tf = torch.from_numpy(map_tf).to(device=depth_map.device, dtype=depth_map.dtype)

#     return map_tf


# def depth_map_to_image(depth_map: torch.Tensor) -> torch.NDArray[torch.uint16]:
#     map_uint16 = (depth_map * 255).cpu().numpy().astype(torch.uint16)
#     return map_uint16
