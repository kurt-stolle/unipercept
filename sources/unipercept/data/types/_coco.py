"""Models the structures in the COCO dataset representation format."""

from __future__ import annotations

from typing import Mapping, NotRequired, Sequence, TypedDict

__all__ = ["COCOAnnotation", "COCOCategory", "COCOImage", "COCOManifest"]


class COCOAnnotation(TypedDict):
    """An annotation, usually part of a ``COCOImage`` definition, following the COCO format."""

    id: int
    category_id: int
    file_name: NotRequired[str]
    iscrowd: NotRequired[int]
    area: NotRequired[int]
    bbox: NotRequired[Sequence[int]]  # default format is XYWH


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
