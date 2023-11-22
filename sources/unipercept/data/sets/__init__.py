"""This module contains the dataset modules."""
from __future__ import annotations

from typing_extensions import deprecated
from unicore import catalog

from . import (
    cityscapes,
    huggingface,
    kitti_360,
    kitti_sem,
    kitti_step,
    pascal_voc,
    vistas,
    wilddash,
)
from ._base import *
from ._pseudo import *

__all__ = []
get_dataset = catalog.get_dataset
get_info = catalog.get_info
list_datasets = catalog.list_datasets
list_info = catalog.list_info


def get_manifest(name, **kwargs):
    """Returns only the manifest of a dataset."""
    return get_dataset(name)(queue_fn=None, **kwargs).manifest


def get_queue(name, **kwargs):
    """Return only the queue of a dataset."""
    return get_dataset(name)(**kwargs).queue


def get_pipeline(name, **kwargs):
    """Returns only the pipeline of a dataset, i.e. the actual loaded images."""
    return get_dataset(name)(**kwargs).pipeline


@deprecated("Registration is no longer required when using `get_{{info,dataset}}` from `unipercept.data.sets`.")
def register():
    import warnings

    # deprecated
    warnings.warn(
        f"Registration not required when using `get_{{info,dataset}}` from `{__name__}`.",
        DeprecationWarning,
        stacklevel=1,
    )


del catalog
