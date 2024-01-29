"""Interface for loading datasets from HuggingFace."""

from __future__ import annotations

import typing as T

import typing_extensions as TX

try:
    import datasets
except ImportError:
    datasets = None

from ._base import PerceptionDataset, info_factory

__all__ = []


class HuggingfaceDataset(PerceptionDataset):
    pass
