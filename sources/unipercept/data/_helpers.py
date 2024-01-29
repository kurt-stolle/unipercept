"""
Tools for working with datasets, especially useful in configuration files and tests.
"""

from __future__ import annotations

import functools
import typing as T

if T.TYPE_CHECKING:
    import unipercept as up

__all__ = ["read_info"]


@functools.lru_cache(maxsize=None)
def read_info(dl: up.data.DataConfig | T.Any, entry="train") -> up.data.sets.Metadata:
    """
    Return metadata about the dataset used for training or evaluation.

    Parameters
    ----------
    dl : DataConfig | T.Any
        The data configuration object.
    entry : str
        The entry point to the dataset (e.g. "train", "val", "test").

    Returns
    -------
    Metadata
        A dictionary containing metadata about the dataset.

    Raises
    ------
    TypeError
        If the dataset class has no `read_info` method.
    """
    from unipercept._lazy import locate

    cls = locate(dl.loaders[entry].dataset._target_)  # type: ignore

    if not hasattr(cls, "read_info"):
        raise TypeError(f"Dataset {cls} has no method read_info!")

    return cls.read_info()  # type: ignore
