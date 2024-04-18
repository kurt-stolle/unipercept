r"""
Utilities for working with (boudning) boxes.
"""

from __future__ import annotations

import enum as E
import typing as T

import torch
from torchvision.tv_tensors import BoundingBoxFormat as _TVBBoxFormat


class BBoxFormat(E.StrEnum):
    """
    Coordinate format of a bounding box.
    """

    LTRB = E.auto()  # (left        , top       , right         , bottom)
    LTWH = E.auto()  # (left        , top       , width         , height)
    CCWH = E.auto()  # (center x    , center y  , width         , height)
    CCAH = E.auto()  # (center x    , center y  , aspect ratio  , height)


def translate_bbox_format_to_torchvision(src: BBoxFormat) -> _TVBBoxFormat:
    """
    Convert a :class:`BBoxFormat` to a :class:`torchvision.tv_tensors.BoundingBoxFormat`.

    Parameters
    ----------
    src : BBoxFormat
        The format to convert.
    Returns
    -------
    torchvision.tv_tensors.BoundingBoxFormat
        The converted format.

    Notes
    -----
    This method was added for completeness after torchvision introduced their own
    bounding box format. Users are encoraged to stick with the `BBoxFormat` wherever
    possible to avoid unnecessary conversions.
    """
    match src:
        case BBoxFormat.LTRB:
            dst = _TVBBoxFormat.XYXY
        case BBoxFormat.LTWH:
            dst = _TVBBoxFormat.XYWH
        case BBoxFormat.CCWH:
            dst = _TVBBoxFormat.CXCYWH
        case _:
            msg = f"Format {src!r} cannot be converted to a torchvision format"
            raise ValueError(msg)
    return dst  # noqa: R504


def translate_bbox_format_from_torchvision(
    src: _TVBBoxFormat,
) -> BBoxFormat:
    """
    Convert a :class:`torchvision.tv_tensors.BoundingBoxFormat` to a :class:`BBoxFormat`.

    Parameters
    ----------
    src : torchvision.tv_tensors.BoundingBoxFormat
        The format to convert.
    Returns
    -------
    BBoxFormat
        The converted format.

    Notes
    -----
    This method was added for completeness after torchvision introduced their own
    bounding box format. Users are encoraged to stick with the `BBoxFormat` wherever
    possible to avoid unnecessary conversions.
    """
    match src:
        case _TVBBoxFormat.XYXY:
            dst = BBoxFormat.LTRB
        case _TVBBoxFormat.XYWH:
            dst = BBoxFormat.LTWH
        case _TVBBoxFormat.CXCYWH:
            dst = BBoxFormat.CCWH
        case _:
            msg = f"Format {src!r} cannot be converted to a BBoxFormat"
            raise ValueError(msg)
    return dst  # noqa: R504


def convert_boxes(
    boxes: torch.Tensor,
    src: BBoxFormat | _TVBBoxFormat,
    dst: BBoxFormat | _TVBBoxFormat,
) -> torch.Tensor:
    """
    Convert a set of bounding boxes from one format to another.

    Parameters
    ----------
    boxes : torch.Tensor
        The boxes to convert.
    src : BBoxFormat
        The source (current) format of :attr:`boxes`.
    dst : BBoxFormat
        The destination (target) format of the converted boxes.

    Returns
    -------
    torch.Tensor
        The converted boxes.
    """
    if isinstance(src, _TVBBoxFormat):
        src = translate_bbox_format_from_torchvision(src)
    if isinstance(dst, _TVBBoxFormat):
        dst = translate_bbox_format_from_torchvision(dst)

    if src == dst:
        return boxes

    # Convert to ltrb
    match src:
        case BBoxFormat.LTWH:
            ltrb = ltwh_to_ltrb(boxes)
        case BBoxFormat.CCWH:
            ltrb = ccwh_to_ltrb(boxes)
        case BBoxFormat.CCAH:
            ltrb = ccah_to_ltrb(boxes)
        case _:
            msg = f"Unsupported conversion from {src!r}"
            raise NotImplementedError(msg)
    match dst:
        case BBoxFormat.LTRB:
            boxes = ltrb
        case BBoxFormat.LTWH:
            boxes = ltrb_to_ltwh(ltrb)
        case BBoxFormat.CCWH:
            boxes = ltrb_to_ccwh(ltrb)
        case BBoxFormat.CCAH:
            boxes = ltrb_to_ccah(ltrb)
        case _:
            msg = f"Unsupported conversion to {dst!r}"
            raise NotImplementedError(msg)

    return boxes  # noqa: R504


def boxes_to_areas(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Compute the areas of a set of boxes.
    """
    if ltrb.numel() == 0:
        return torch.zeros(ltrb.shape[:-1], device=ltrb.device, dtype=ltrb.dtype)

    w = ltrb[..., 2] - ltrb[..., 0]
    h = ltrb[..., 3] - ltrb[..., 1]

    return w * h


def scale_boxes(
    ltrb: torch.Tensor,
    scale: torch.Tensor | torch.Size | int | float | T.Iterable[int | float],
) -> torch.Tensor:
    """
    Scale a set of boxes by a fixed ratio.

    Parameters
    ----------
    ltrb : torch.Tensor
        The boxes to scale, in [x1, y1, x2, y2] format.
    scale : Union[torch.Tensor, torch.Size, float, Sequence[float]]
        The scale factor(s) to apply to the boxes, when a single value is provided, it is
        used for both dimensions.
    Returns
    -------
    torch.Tensor
        The scaled boxes.
    """
    if ltrb.numel() == 0:
        return ltrb

    if isinstance(scale, (torch.Tensor)):
        sw, sh = map(float, scale.tolist())
    elif not isinstance(scale, T.Iterable):
        sw = sh = float(scale)
    else:
        sw, sh = map(float, scale)

    return torch.stack(
        [
            ltrb[..., 0] * sw,
            ltrb[..., 1] * sh,
            ltrb[..., 2] * sw,
            ltrb[..., 3] * sh,
        ],
        dim=-1,
    ).to(ltrb.dtype)


def reposition_boxes_in_crop(
    boxes_ltrb: torch.Tensor, crop_ltrb: torch.Tensor, clamp: bool = False
) -> torch.Tensor:
    """
    Reposition a set of boxes that have been defined relative to some image size to be
    relative to a crop of that image.

    Parameters
    ----------
    boxes_ltrb : torch.Tensor
        The boxes to reposition, in [x1, y1, x2, y2] format.
    crop_ltrb : torch.Tensor
        The crop to reposition the boxes into, in [x1, y1, x2, y2] format.
    clamp : bool
        Whether to clamp the boxes to the crop boundaries.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """

    if boxes_ltrb.numel() == 0:
        return boxes_ltrb

    boxes_x1 = boxes_ltrb[..., 0]
    boxes_y1 = boxes_ltrb[..., 1]
    boxes_x2 = boxes_ltrb[..., 2]
    boxes_y2 = boxes_ltrb[..., 3]

    crop_x1 = crop_ltrb[..., 0]
    crop_y1 = crop_ltrb[..., 1]
    crop_x2 = crop_ltrb[..., 2]
    crop_y2 = crop_ltrb[..., 3]

    if clamp:
        boxes_x1 = torch.clamp(boxes_x1, min=crop_x1)
        boxes_y1 = torch.clamp(boxes_y1, min=crop_y1)
        boxes_x2 = torch.clamp(boxes_x2, max=crop_x2)
        boxes_y2 = torch.clamp(boxes_y2, max=crop_y2)

    boxes_x1 -= crop_x1
    boxes_y1 -= crop_y1
    boxes_x2 -= crop_x1
    boxes_y2 -= crop_y1

    return torch.stack([boxes_x1, boxes_y1, boxes_x2, boxes_y2], dim=-1)


def expand_boxes_to_squares(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Pad a set of boxes such that they are square.
    """
    if ltrb.numel() == 0:
        return ltrb

    w = ltrb[..., 2] - ltrb[..., 0]
    h = ltrb[..., 3] - ltrb[..., 1]

    max_size = torch.max(w, h)
    half_diff = (max_size - torch.stack([w, h], dim=-1)) / 2

    return torch.stack(
        [
            ltrb[..., 0] - half_diff[..., 0],
            ltrb[..., 1] - half_diff[..., 1],
            ltrb[..., 2] + half_diff[..., 0],
            ltrb[..., 3] + half_diff[..., 1],
        ],
        dim=-1,
    )


def expand_boxes_by_pixels(ltrb: torch.Tensor, px: int) -> torch.Tensor:
    """
    Expand a set of boxes by a fixed number of pixels.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ltrb.numel() == 0:
        return ltrb

    return torch.stack(
        [
            ltrb[..., 0] - px,
            ltrb[..., 1] - px,
            ltrb[..., 2] + px,
            ltrb[..., 3] + px,
        ],
        dim=-1,
    )


def expand_boxes_by_ratio(ltrb: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    Expand a set of boxes by a fixed ratio.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ltrb.numel() == 0:
        return ltrb

    w = ltrb[..., 2] - ltrb[..., 0]
    h = ltrb[..., 3] - ltrb[..., 1]

    half_diff_w = (w * ratio) / 2
    half_diff_h = (h * ratio) / 2

    return torch.stack(
        [
            ltrb[..., 0] - half_diff_w,
            ltrb[..., 1] - half_diff_h,
            ltrb[..., 2] + half_diff_w,
            ltrb[..., 3] + half_diff_h,
        ],
        dim=-1,
    )


def ltrb_to_ltwh(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ltrb.numel() == 0:
        return ltrb

    w = ltrb[..., 2] - ltrb[..., 0]
    h = ltrb[..., 3] - ltrb[..., 1]
    x = ltrb[..., 0]
    y = ltrb[..., 1]

    return torch.stack([x, y, w, h], dim=-1)


def ltwh_to_ltrb(ltwh: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ltwh.numel() == 0:
        return ltwh

    x1 = ltwh[..., 0]
    y1 = ltwh[..., 1]
    x2 = x1 + ltwh[..., 2]
    y2 = y1 + ltwh[..., 3]

    return torch.stack([x1, y1, x2, y2], dim=-1)


def ltrb_to_ccwh(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ltrb.numel() == 0:
        return ltrb

    cx = (ltrb[..., 0] + ltrb[..., 2]) / 2
    cy = (ltrb[..., 1] + ltrb[..., 3]) / 2
    w = ltrb[..., 2] - ltrb[..., 0]
    h = ltrb[..., 3] - ltrb[..., 1]

    return torch.stack([cx, cy, w, h], dim=-1)


def ccwh_to_ltrb(ccwh: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ccwh.numel() == 0:
        return ccwh

    x1 = ccwh[..., 0] - ccwh[..., 2] / 2
    y1 = ccwh[..., 1] - ccwh[..., 3] / 2
    x2 = ccwh[..., 0] + ccwh[..., 2] / 2
    y2 = ccwh[..., 1] + ccwh[..., 3] / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


def ltrb_to_ccah(ltrb: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [cx, cy, ar, h] format.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ltrb.numel() == 0:
        return ltrb

    cx = (ltrb[..., 0] + ltrb[..., 2]) / 2
    cy = (ltrb[..., 1] + ltrb[..., 3]) / 2
    ar = (ltrb[..., 2] - ltrb[..., 0]) / (ltrb[..., 3] - ltrb[..., 1])
    h = ltrb[..., 3] - ltrb[..., 1]

    return torch.stack([cx, cy, ar, h], dim=-1)


def ccah_to_ltrb(ccah: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [cx, cy, ar, h] to [x1, y1, x2, y2] format.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ccah.numel() == 0:
        return ccah

    x1 = ccah[..., 0] - ccah[..., 2] / 2
    y1 = ccah[..., 1] - ccah[..., 3] / 2
    x2 = ccah[..., 0] + ccah[..., 2] / 2
    y2 = ccah[..., 1] + ccah[..., 3] / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


def ccwh_to_ccah(ccwh: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [cx, cy, w, h] to [cx, cy, ar, h] format.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ccwh.numel() == 0:
        return ccwh

    ar = ccwh[..., 2] / ccwh[..., 3]

    return torch.stack([ccwh[..., 0], ccwh[..., 1], ar, ccwh[..., 3]], dim=-1)


def ccah_to_ccwh(ccah: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [cx, cy, ar, h] to [cx, cy, w, h] format.

    Notes
    -----
    This function was generated using GitHub Copilot.
    """
    if ccah.numel() == 0:
        return ccah

    w = ccah[..., 3] * ccah[..., 2]

    return torch.stack([ccah[..., 0], ccah[..., 1], w, ccah[..., 3]], dim=-1)
