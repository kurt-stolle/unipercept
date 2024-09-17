r"""
Generic pattern-based dataset.
"""

from __future__ import annotations

import typing as T

import regex as re
import typing_extensions as TX

from . import Metadata, PerceptionDataset

if T.TYPE_CHECKING:
    from ..tensors import DepthFormat, LabelsFormat


class PatternDataset(PerceptionDataset, id=None, info=None, use_manifest_cache=False):
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
