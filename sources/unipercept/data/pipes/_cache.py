from __future__ import annotations

import os
import typing as T
import warnings
from typing import Generic, Iterable, Iterator, Mapping, TypeVar

import torch

from unipercept import file_io

__all__ = ["LazyPickleCache"]

_M = T.TypeVar("_M", bound=dict)


class LazyPickleCache(Generic[_M]):
    def __init__(self, path: str | os.PathLike):
        self.path = str(file_io.Path(path))

    @staticmethod
    def store(path: str | os.PathLike, items: _M) -> None:
        items = dict(items)  # type: ignore

        path = file_io.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            torch.save(items, fh)  # type: ignore

    @property
    # @functools.cached_property
    def data(self) -> _M:
        path = file_io.Path(self.path, force=True)
        with path.open("rb") as fh:
            items = torch.load(fh)  # type: ignore

        assert isinstance(items, dict), type(items)
        return T.cast(_M, items)

    @data.setter
    def data(self, items: _M) -> None:
        self.store(self.path, items)

    def __len__(self) -> int:
        return len(self.data)
