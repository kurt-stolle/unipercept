"""Helpers for working with registration-based datasets in UniPercept."""

from __future__ import annotations

import importlib

from unicore.catalog import get_dataset, get_info, list_datasets, list_info

__all__ = ["get_info", "get_dataset", "list_datasets", "list_info", "get_manifest", "get_queue", "get_pipeline"]


def get_manifest(name, **kwargs):
    """Returns only the manifest of a dataset."""
    return get_dataset(name)(queue_fn=None, **kwargs).manifest


def get_queue(name, **kwargs):
    """Return only the queue of a dataset."""
    return get_dataset(name)(**kwargs).queue


def get_pipeline(name, **kwargs):
    """Returns only the pipeline of a dataset, i.e. the actual loaded images."""
    return get_dataset(name)(**kwargs).pipeline
