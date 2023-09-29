"""
Tools for working with datasets, especially useful in configuration files and tests.
"""

import functools
import typing as T

from ._config import DataConfig
from .types import Metadata

__all__ = ["read_info"]


@functools.lru_cache(maxsize=None)
def read_info(dl: DataConfig | T.Any, entry="train") -> Metadata:
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
    from uniutils.config._lazy import locate

    cls = locate(dl.loaders[entry].dataset._target_)  # type: ignore

    if not hasattr(cls, "read_info"):
        raise TypeError(f"Dataset {cls} has no method read_info!")

    return cls.read_info()  # type: ignore
