import enum as E
import typing as T

import safetensors.torch
import torch
from torchvision.tv_tensors import Mask as _Mask

from unipercept import file_io
from unipercept.data.tensors.helpers import read_pixels
from unipercept.types import Tensor

from .registry import pixel_maps

__all__ = ["Mask", "MaskFormat", "read_mask", "save_mask"]


@pixel_maps.register
class Mask(_Mask):
    pass


class MaskFormat(E.StrEnum):
    PNG_L = E.auto()
    PNG_LA = E.auto()
    PNG_L16 = E.auto()
    PNG_LA16 = E.auto()
    TORCH = E.auto()
    SAFETENSORS = E.auto()


class MaskMeta(T.TypedDict):
    format: T.NotRequired[MaskFormat]


def read_mask(path: file_io.Pathable, /, **kwargs: MaskMeta) -> Mask:
    format = kwargs["format"]
    path = file_io.get_local_path(path)

    match format:
        case MaskFormat.PNG_L | MaskFormat.PNG_L16:
            return read_pixels(path, color=False, alpha=False).as_subclass(Mask)
        case MaskFormat.PNG_LA | MaskFormat.PNG_LA16:
            return read_pixels(path, color=False, alpha=True).as_subclass(Mask)
        case MaskFormat.TORCH:
            return torch.load(path).as_subclass(Mask)
        case MaskFormat.SAFETENSORS:
            return safetensors.torch.load(path).as_subclass(Mask)
        case _:
            msg = f"Unsupported format: {format}"
            raise NotImplementedError(msg)


def save_mask(
    tensor: Tensor | Mask, path: file_io.Pathable, /, **kwargs: MaskMeta
) -> Mask:
    raise NotImplementedError("Not implemented yet.")
