from __future__ import annotations

import typing as T
from enum import StrEnum, auto

import cv2
import PIL.Image as pil_image
import safetensors.torch as safetensors
import torch
from torch.types import Device
from torchvision.transforms.v2.functional import resize_image, register_kernel, InterpolationMode
from torchvision.transforms.v2.functional._geometry import _compute_resized_output_size
from torchvision.tv_tensors import Mask
from torch import Tensor
from einops import rearrange

from unipercept.data.tensors.helpers import get_kwd, read_pixels, write_png_l16
from unipercept.data.tensors.registry import pixel_maps
from unipercept.utils.typings import Pathable

__all__ = ["DepthMap", "DepthFormat", "downsample_depthmap", "resize_depthmap"]

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
        import numpy as np

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
            case DepthFormat.DEPTH_INT16:
                # depth_image = (self * float(2**8)).numpy().astype(np.uint16)
                # image = pil_image.fromarray(depth_image, mode="I;16")
                # image.save(path)
                write_png_l16(path, self * float(2**8))

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


def downsample_depthmap(x: Tensor, size: tuple[int, int]) -> Tensor:
    """
    Downsampling of depth maps via median pooling.
    """
    x = rearrange(x, "b (h1 h2) (w1 w2) -> b h1 w1 (h2 w2)", h1=size[0], w1=size[1])
    x[x <= 0] = torch.nan
    x = torch.nanmedian(x, dim=-1).values
    x[~torch.isfinite(x)] = 0

    return x

@register_kernel(functional="resize", tv_tensor_cls=DepthMap)
def resize_depthmap(
    image: DepthMap,
    size: T.List[int],
    interpolation: T.Any = None,  # noqa: U100
    max_size: int | None = None,
    antialias: T.Any = True,  # noqa: U100
    use_rescale: bool = False,
) -> torch.Tensor:
    shape = image.shape
    h_old, w_old = shape[-2:]
    h_new, w_new = _compute_resized_output_size((h_old, w_old), size=size, max_size=max_size)

    if h_new <= h_old and w_new <= w_old:
        res = downsample_depthmap(image, (h_new, w_new))
    else:
        res = resize_image(image, size, interpolation=InterpolationMode.NEAREST_EXACT, max_size=max_size, antialias=antialias)
    
    if use_rescale:
        d_min = image.min()
        d_max = image.max()
        scale = (h_old / h_new + w_old / w_new) / 2

        res = (res * scale).clamp(d_min, d_max)

    return res.as_subclass(DepthMap)
