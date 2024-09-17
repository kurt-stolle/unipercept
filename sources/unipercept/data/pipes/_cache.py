from __future__ import annotations

import os
import pickle
import typing as T
import warnings
from typing import Generic

import yaml

from unipercept import file_io

__all__ = ["LazyPickleCache", "LazyYAMLCache"]

_M = T.TypeVar("_M", bound=dict)


class LazyPickleCache(Generic[_M]):
    """
    A cache that reads and writes pickle files.
    """

    def _read_file(self, fh: T.IO) -> _M:
        return pickle.load(fh)

    def _write_file(self, items: _M, fh: T.IO) -> None:
        pickle.dump(items, fh)

    is_binary: T.ClassVar = True
    file_ext: T.ClassVar = ".pkl"

    def __init__(self, path: str):
        self.path = path
        assert self.path.endswith(self.file_ext), self.path

    def exists(self, *, check_valid: bool = True) -> bool:
        path = file_io.get_local_path(self.path)
        if not os.path.exists(path) or not os.path.isfile(path):
            return False
        if check_valid:
            try:
                with open(path, "rb" if self.is_binary else "r") as fh:
                    items = self._read_file(fh)  # type: ignore
                return isinstance(items, dict)
            except Exception as e:
                warnings.warn(f"Cac: {e}")
                return False
        return True

    def store(self, items: _M) -> None:
        items = dict(items)  # type: ignore
        path = file_io.Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb" if self.is_binary else "w") as fh:
            self._write_file(items, fh)  # type: ignore

    def read(self) -> _M:
        path = file_io.get_local_path(self.path, force=True)
        with open(path, "rb" if self.is_binary else "r") as fh:
            items = self._read_file(fh)  # type: ignore

        assert isinstance(items, dict), type(items)
        return T.cast(_M, items)


class LazyYAMLCache(LazyPickleCache, Generic[_M]):
    """
    A cache that reads and writes YAML files.
    """

    def _read_file(self, fh: T.IO) -> _M:
        return yaml.unsafe_load(fh)

    def _write_file(self, items: _M, fh: T.IO) -> None:
        yaml.dump(items, fh)

    write_fn = yaml.dump
    is_binary = False
    file_ext = ".yaml"
