from __future__ import annotations

import typing as T
from enum import StrEnum, auto

import PIL.Image as pil_image
import safetensors.torch as safetensors
import torch
from torch.types import Device
from torchvision.tv_tensors import Mask

from unipercept.data.tensors.helpers import get_kwd, read_pixels
from unipercept.data.tensors.registry import pixel_maps
from unipercept.utils.typings import Pathable

__all__ = ["DepthMap", "DepthFormat"]

DEFAULT_DEPTH_DTYPE: T.Final = torch.float32


class DepthFormat(StrEnum):
    TIFF = auto()
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

    def save(self, path: Pathable, format: DepthFormat | str | None = None) -> None:
        from unipercept import file_io

        path = file_io.Path(path)
        if format is None:
            match path.suffix.lower():
                case ".tiff":
                    format = DepthFormat.TIFF
                case ".pth", ".pt":
                    format = DepthFormat.TORCH
                case ".safetensors":
                    format = DepthFormat.SAFETENSORS
                case _:
                    msg = f"Could not infer depth format from path: {path}"
                    raise ValueError(msg)

        path.parent.mkdir(parents=True, exist_ok=True)

        match DepthFormat(format):
            case DepthFormat.TIFF:
                depth_image = pil_image.fromarray(
                    self.float().squeeze_(0).cpu().numpy()
                )
                if depth_image.mode != "F":
                    msg = f"Expected image format 'F'; Got {depth_image.mode!r}"
                    raise ValueError(msg)
                depth_image.save(path, format="TIFF")
            case DepthFormat.SAFETENSORS:
                safetensors.save_file({"data": torch.as_tensor(self)}, path)
            case DepthFormat.TORCH:
                torch.save(torch.as_tensor(self), path)
            case _:
                msg = f"Unsupported depth format: {format}"
                raise NotImplementedError(msg)

    @classmethod
    @torch.no_grad()
    def read(
        cls,
        path: Pathable,
        dtype: torch.dtype = DEFAULT_DEPTH_DTYPE,
        **meta_kwds: T.Any,
    ) -> T.Self:
        import numpy as np

        from unipercept import file_io

        path = file_io.get_local_path(str(path))
        # Switch by depth format
        format = get_kwd(meta_kwds, "format", DepthFormat | str)
        match DepthFormat(format):  # type: ignore
            case DepthFormat.TIFF:
                m = pil_image.open(path)
                if m.mode != "F":
                    msg = f"Expected image format 'F'; Got {m.format!r}"
                    raise ValueError(msg)
                m = torch.from_numpy(np.array(m, copy=True))
            case DepthFormat.DEPTH_INT16:
                m = read_pixels(path, color=False) / float(2**8)
            case DepthFormat.DISPARITY_INT16:
                m = cls.read_from_disparity(path, **meta_kwds)
            case DepthFormat.SAFETENSORS:
                m = safetensors.load_file(path)["data"]
            case DepthFormat.TORCH:
                m = torch.load(path, map_location="cpu").squeeze_(0).squeeze_(0)
            case _:
                msg = f"Unsupported depth format: {format}"
                raise NotImplementedError(msg)

        # TODO: Add angular FOV compensation via metadata
        m = m.to(dtype=dtype)
        m[m == torch.inf] = 0.0
        m[m == torch.nan] = 0.0
        m.squeeze_()
        if m.ndim > 2:
            msg = f"Depth map has {m.ndim} dimensions, expected 2"
            raise ValueError(msg)

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
