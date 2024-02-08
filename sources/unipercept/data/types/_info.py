"""Generic descriptive types for data."""

from __future__ import annotations

import typing as T
from pathlib import Path
from typing import Any, TypeAlias

from unipercept.utils.frozendict import frozendict

__all__ = [
    "HW",
    "BatchType",
    "ImageSize",
]

BatchType: TypeAlias = list[frozendict[str, Any]]
HW: TypeAlias = tuple[int, int]
PathType: TypeAlias = Path | str
OptionalPath: TypeAlias = PathType | None


class ImageSize(T.NamedTuple):
    height: int
    width: int


class SampleInfo(T.TypedDict, total=False):
    num_instances: frozendict[int, int]  # Mapping: (Dataset ID) -> (Num. instances)
    num_pixels: frozendict[int, int]  # Mapping: (Dataset ID) -> (Num. pixels)
