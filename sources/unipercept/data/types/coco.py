"""Models the structures in the COCO dataset representation format.

See Also
--------

- Data format: https://cocodataset.org/#format-data
- Results format: https://cocodataset.org/#format-results
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NotRequired, TypedDict

__all__ = ["COCOAnnotation", "COCOCategory", "COCOImage", "COCOManifest"]


class COCOResultObjectDetectionBBox(TypedDict):
    r"""An object detection result, following the COCO format."""

    image_id: int
    category_id: int
    score: float
    bbox: tuple[float, float, float, float]  # XYXY from top-left corner


class COCOResultObjectDetectionSegmentation(TypedDict):
    r"""An object detection result, following the COCO format."""

    image_id: int
    category_id: int
    score: float
    segmentation: list[list[float]]  # RLE


class COCOResultKeypoint(TypedDict):
    r"""A keypoint result, following the COCO format."""

    image_id: int
    category_id: int
    score: float
    keypoints: list[float]  # [x1, y1, v1, x2, y2, v2, ...]


class COCOResultPanopticSegment(TypedDict):
    r"""A single segment's metadata as part of a panoptic segmentation result, following the COCO format."""

    id: int
    category_id: int


class COCOResultPanoptic(TypedDict):
    r"""A panoptic segmentation result, following the COCO format."""

    image_id: int
    file_name: str
    segments_info: list[COCOResultPanopticSegment]


class COCOAnnotation(TypedDict):
    r"""An annotation, usually part of a ``COCOImage`` definition, following the COCO format."""

    id: int
    category_id: int
    file_name: NotRequired[str]
    iscrowd: NotRequired[int]
    area: NotRequired[int]
    bbox: NotRequired[Sequence[int]]  # XYXY from top-left corner
    segmentation: NotRequired[Sequence[Sequence[int]]]  #  RLE


class COCOCategory(TypedDict):
    """A category ("class") in the dataset, following the COCO format."""

    color: Sequence[int]  # default format is RGB
    isthing: NotRequired[int]
    id: int
    trainId: int
    name: str


class COCOImage(TypedDict):
    """An image in a manifest, following the COCO format."""

    id: int
    file_name: str
    height: int
    width: int
    annotations: Sequence[COCOAnnotation]


class COCOManifest(TypedDict):
    """The manifest, e.g. the standard format of a COCO-style JSON document."""

    images: Sequence[COCOImage]
    categories: Sequence[COCOCategory]
    annotations: Sequence[COCOAnnotation]
    info: NotRequired[Mapping[str, str]]
    licenses: NotRequired[Sequence[Mapping[str, str]]]
