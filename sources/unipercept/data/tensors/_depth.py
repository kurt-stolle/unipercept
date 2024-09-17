from __future__ import annotations

import enum as E
import typing as T

import PIL.Image as pil_image
import safetensors.torch as safetensors
import torch
from einops import rearrange
from torch import Tensor
from torch.nn.functional import interpolate
from torch.types import Device
from torchvision.transforms.v2.functional import register_kernel
from torchvision.transforms.v2.functional._geometry import _compute_resized_output_size
from torchvision.tv_tensors import Mask

from unipercept.data.tensors.helpers import get_kwd, read_pixels, write_png_l16
from unipercept.data.tensors.registry import pixel_maps
from unipercept.types import Pathable

__all__ = [
    "DepthMap",
    "DepthMode",
    "DepthFormat",
    "downsample_depthmap",
    "resize_depthmap",
    "absolute_to_normalized_depth",
    "normalized_to_absolute_depth",
]

DEFAULT_DEPTH_DTYPE: T.Final = torch.float32


class DepthFormat(E.StrEnum):
    r"""
    Enum class for depth map file formats and their respective mode.
    """

    TIFF = E.auto()
    DEPTH_INT16 = E.auto()
    DISPARITY_INT16 = E.auto()
    TORCH = E.auto()
    SAFETENSORS = E.auto()


class DepthMode(E.StrEnum):
    r"""
    Enum class for depth prediction modes.
    """

    ABSOLUTE = E.auto()
    DISPARITY = E.auto()


@pixel_maps.register
class DepthMap(Mask):
    def __new__(
        cls,
        data: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str, int]] = None,
        requires_grad: Optional[bool] = None,
    ) -> T.Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        return tensor.as_subclass(cls)

    @classmethod
    def default_like(cls, other: Tensor) -> T.Self:
        """Returns a default instance of this class with the same shape as the given tensor."""
        return cls(torch.full_like(other, fill_value=0, dtype=torch.float32))

    @classmethod
    def default(cls, shape: torch.Size, device: Device = "cpu") -> T.Self:
        """Returns a default instance of this class with the given shape."""
        return cls(torch.zeros(shape, device=device, dtype=torch.float32))  # type: ignore

    def save(self, path: Pathable, format: DepthFormat | str | None = None) -> None:
        from unipercept import file_io

        self = self.detach().cpu()

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
                assert path.suffix.lower() == ".tiff", path
                depth_image.save(path, format="TIFF")
            case DepthFormat.SAFETENSORS:
                assert path.suffix.lower() == ".safetensors", path
                safetensors.save_file({"data": torch.as_tensor(self)}, path)
            case DepthFormat.TORCH:
                torch.save(torch.as_tensor(self), path)
            case DepthFormat.DEPTH_INT16:
                # depth_image = (self * float(2**8)).numpy().astype(np.uint16)
                # image = pil_image.fromarray(depth_image, mode="I;16")
                # image.save(path)
                assert path.suffix.lower() == ".png", path
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


class DepthDownsampleMethod(E.StrEnum):
    MEDIAN = E.auto()
    NEAREST = E.auto()


def downsample_depthmap(
    depth_map: Tensor,
    size: tuple[int, int] | torch.Size,
    method: DepthDownsampleMethod | str = DepthDownsampleMethod.MEDIAN,
) -> Tensor:
    """
    Downsampling of depth maps.

    Parameters
    ----------
    depth_map: Tensor[..., H_old, W_old]
        The input depth map.
    size : tuple[H_new, W_new]
        The target size of the downsampled depth map.
    method : DepthDownsampleMethod or str
        The method used for downsampling. Default: ```DepthDownsampleMethod.MEDIAN``.

    Returns
    -------
    Tensor[..., H, W]
        The downsampled depth map.
    """
    # Check that the shape is actually divisible by the target size, else do NN downsample first to the closest size
    h_old, w_old = depth_map.shape[-2:]
    h_new, w_new = size

    match DepthDownsampleMethod(method):
        case DepthDownsampleMethod.MEDIAN:
            if h_old % h_new != 0 or w_old % w_new != 0:
                h_new = h_old // round(h_old / h_new)
                w_new = w_old // round(w_old / w_new)
                depth_map = interpolate_depthmap(depth_map, (h_new, w_new))

            # Perform median pooling
            depth_map = rearrange(
                depth_map,
                "... (h1 h2) (w1 w2) -> ... h1 w1 (h2 w2)",
                h1=h_new,
                w1=w_new,
            )
            depth_map[depth_map <= 0] = torch.nan
            depth_map = torch.nanmedian(depth_map, dim=-1).values

            # Set invalid depth values to 0
            depth_map[~torch.isfinite(depth_map)] = 0
        case DepthDownsampleMethod.NEAREST:
            depth_map = interpolate_depthmap(depth_map, size)

    return depth_map


@register_kernel(functional="resize", tv_tensor_cls=DepthMap)
def resize_depthmap(
    image: DepthMap,
    size: list[int],
    interpolation: T.Any = None,  # noqa: U100
    max_size: int | None = None,
    antialias: T.Any = False,  # noqa: U100
    use_rescale: bool = False,
) -> Tensor:
    shape = image.shape
    h_old, w_old = shape[-2:]
    h_new, w_new = _compute_resized_output_size(
        (h_old, w_old), size=size, max_size=max_size
    )

    if h_new <= h_old and w_new <= w_old:
        res = downsample_depthmap(image, (h_new, w_new))
    else:
        res = interpolate_depthmap(image, (h_new, w_new))

    if use_rescale:
        d_min = image.min()
        d_max = image.max()
        scale = (h_old / h_new + w_old / w_new) / 2

        res = (res * scale).clamp(d_min, d_max)

    return res.as_subclass(DepthMap)


def interpolate_depthmap(
    depth_map: Tensor,
    size: tuple[int, int] | torch.Size,
) -> Tensor:
    """
    Quick wrapper for 2D nearest neighbor interpolation, if the input does not have enough
    dimensions, then these are added before interpolation and removed at the end.
    """
    ndim = depth_map.ndim
    while depth_map.ndim < 4:
        depth_map = depth_map.unsqueeze(0)
    depth_map = interpolate(depth_map, size=size, mode="nearest-exact")
    while depth_map.ndim > ndim:
        depth_map = depth_map.squeeze(0)

    return depth_map


def clamp_absolute_depth(
    value: Tensor,
    min_depth: float | Tensor,
    max_depth: float | Tensor,
) -> Tensor:
    """
    Clamp depth values to the specified range. Uses ReLU instead of ``torch.clamp``
    to avoid backpropagation through the gradient of the clamp function, which results
    in numerical inaccuracy for PyTorch version < 2.3.0.

    Parameters
    ----------
    value : Tensor[..., N, H, W]
        The depth tensor.
    min_depth : float or Tensor[..., N]
        The minimum depth value.
    max_depth : float or Tensor[..., N]
        The maximum depth value.

    Returns
    -------
    Tensor[..., N, H, W]
        The clamped depth tensor.
    """
    result = (value - min_depth).relu() + min_depth
    result = max_depth - (max_depth - result).relu()
    return result


def normalized_to_absolute_depth(
    value: Tensor,
    min_depth: float | Tensor,
    max_depth: float | Tensor,
    mode: DepthMode | str = DepthMode.ABSOLUTE,
) -> Tensor:
    """
    Convert depth from normalized values to absolute range.

    Notes
    -----
    The range of values in the input is not strictly enfoced. Users should ensure
    that the input values are within the normalized range.

    Parameters
    ----------
    value : Tensor[..., N, H, W] in (0, 1)
        The normalized depth tensor.
    min_depth : float or Tensor[..., N]
        The minimum depth value.
    max_depth : float or Tensor[..., N]
        The maximum depth value.
    mode : DepthMode
        The depth prediction mode.

    Returns
    -------
    Tensor[..., N, H, W] in (min_depth, max_depth)
        The absolute depth tensor.
    """
    if mode == DepthMode.ABSOLUTE:
        result = value * (max_depth - min_depth) + min_depth
    elif mode == DepthMode.DISPARITY:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * value
        result = 1 / scaled_disp
    else:
        msg = f"Invalid prediction mode: {mode}"
        raise NotImplementedError(msg)
    result = torch.nan_to_num(result, nan=0, posinf=0, neginf=0)
    return result


def absolute_to_normalized_depth(
    value: Tensor,
    min_depth: float | Tensor,
    max_depth: float | Tensor,
    mode: DepthMode | str,
) -> Tensor:
    """
    Convert depth from absolute range to normalized values.

    Notes
    -----
    The range of values in the input is not strictly enfoced. Users should ensure
    that the input values are within the absolute range.

    Parameters
    ----------
    value : Tensor[..., N, H, W] in (min_depth, max_depth)
        The absolute depth tensor.
    min_depth : float or Tensor[..., N]
        The minimum depth value.
    max_depth : float or Tensor[..., N]
        The maximum depth value.
    mode : DepthMode
        The depth prediction mode.

    Returns
    -------
    Tensor[..., N, H, W] in (0, 1)
        The absolute depth tensor.
    """
    if mode == DepthMode.ABSOLUTE:
        result = (value - min_depth) / (max_depth - min_depth)
    elif mode == DepthMode.DISPARITY:
        min_disp = 1 / max_depth
        max_disp = 1 / min_depth
        scaled_disp = 1 / value
        result = (scaled_disp - min_disp) / (max_disp - min_disp)
    else:
        msg = f"Invalid prediction mode: {mode}"
        raise NotImplementedError(msg)
    result = torch.nan_to_num(result, nan=0, posinf=0, neginf=0)
    return result
