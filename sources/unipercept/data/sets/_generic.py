"""
Defines generic dataset classses.
"""

from __future__ import annotations

import re
import typing as T

import typing_extensions as TX

from . import Metadata, PerceptionDataset

if T.TYPE_CHECKING:
    from ..tensors import DepthFormat, LabelsFormat


__all__ = ["GenericPatternDataset", "GenericCOCODataset"]


class GenericPatternDataset(
    PerceptionDataset, id=None, info=None, use_manifest_cache=False
):
    """A generic dataset that matches input and target patterns."""

    root: str
    like: str
    pattern: re.Pattern
    depth_format: DepthFormat
    depth_path: T.Callable[..., str]
    image_path: T.Callable[..., str]
    panoptic_format: LabelsFormat
    panoptic_path: T.Callable[..., str]

    @TX.override
    def read_info(self) -> Metadata:
        from . import catalog

        return catalog.get_info(self.like)


class GenericCOCODataset(
    PerceptionDataset, id=None, info=None, use_manifest_cache=False
):
    """A generic dataset that uses COCO-like dataset JSON files."""

    path: str
    like: str

    @TX.override
    def read_info(self) -> Metadata:
        from . import catalog

        return catalog.get_info(self.like)
