"""Interface for loading datasets from HuggingFace."""

from __future__ import annotations

try:
    import datasets
except ImportError:
    datasets = None

from unipercept.data.sets._base import PerceptionDataset

__all__ = []

# TODO
