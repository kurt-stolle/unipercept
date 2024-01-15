import os
from typing import Generic, Iterable, Iterator, Mapping, TypeVar

import torch
from typing_extensions import override

from unicore import file_io

__all__ = ["LazyPickleCache"]

_K = TypeVar("_K")
_V = TypeVar("_V")


class LazyPickleCache(Generic[_K, _V]):
    def __init__(self, path: str | os.PathLike):
        self.path = path

    @staticmethod
    def store(path: str | os.PathLike, items: Mapping[_K, _V] | Iterable[tuple[_K, _V]]) -> None:
        if not (isinstance(items, Iterable) or isinstance(items, Mapping)):
            raise TypeError(f"{type(items)} is not a Mapping or Iterable")

        items = dict(items)

        path = file_io.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            torch.save(items, fh)  # type: ignore

    @property
    # @functools.cached_property
    def data(self) -> dict[_K, _V]:
        path = file_io.Path(self.path, force=True)
        with path.open("rb") as fh:
            items = torch.load(fh)  # type: ignore
        assert isinstance(items, dict), type(items)

        return items

    @data.setter
    def data(self, items: Mapping[_K, _V] | Iterable[tuple[_K, _V]]) -> None:
        self.store(self.path, items)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem___(self, index: _K) -> _V:
        return self.data[index]

    def __iter__(self) -> Iterator[tuple[_K, _V]]:
        yield from self.data.items()
