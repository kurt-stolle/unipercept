from __future__ import annotations

import enum as E

import torch

from unipercept.types import Device, DType, Size, Tensor


class GridMode(E.StrEnum):
    INDEX = E.auto()
    NORMALIZED = E.auto()
    PIXEL_CENTER = E.auto()
    PIXEL_NOISE = E.auto()


@torch.no_grad()
def generate_coord_grid(
    canvas_size: Size | tuple[int, int],
    device: Device | None = None,
    dtype: DType | None = None,
    mode: GridMode = GridMode.INDEX,
) -> Tensor:
    r"""
    Generate pixel coordinates grid.

    Parameters
    ----------
    canvas_size:
        Size of the canvas.
    device:
        Device to use.
    dtype:
        Data type to use.
    mode:
        Grid mode to use. By default, `GridMode.INDEX`.

    Returns
    -------
    Tensor[H, W, 2]
        Pixel coordinates grid.
    """
    height, width = canvas_size
    if dtype is None:
        dtype = torch.float32

    if mode == GridMode.NORMALIZED:
        x_range = torch.linspace(-1, 1, width, device=device, dtype=dtype)
        y_range = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    else:
        x_range = torch.arange(width, device=device, dtype=dtype)
        y_range = torch.arange(height, device=device, dtype=dtype)

    coords = torch.stack(
        torch.meshgrid(
            [x_range, y_range],  # (H, W)
            indexing="xy",
        ),
        dim=-1,
    )  # (H, W, 2)
    match mode:
        case GridMode.NORMALIZED | GridMode.INDEX:
            pass
        case GridMode.PIXEL_CENTER:
            coords = coords + 0.5
        case GridMode.PIXEL_NOISE:
            coords += torch.rand_like(coords)
        case _:
            msg = f"Invalid {mode=}. Choose from {list(GridMode)}"
            raise ValueError(msg)
    return coords
