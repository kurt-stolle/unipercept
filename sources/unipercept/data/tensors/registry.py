"""
For use in data operations (see ``sources/unipercept/data/ops.py``).
"""

from __future__ import annotations

import typing as T

_T = T.TypeVar("_T")

__all__ = ["pixel_maps"]


class TypeRegistry:
    """
    Simple registry class to keep track of registered types, e.g. for getting the set of all
    classes that represent image (pixels by pixes) data.
    """

    _name: T.Final[str]
    _set: T.Final[set[type]]

    def __init__(self, name: str) -> None:
        self._name = name
        self._set = set()

    def __repr__(self) -> str:
        items = ", ".join(map(repr, self._set))
        return f"<{self.__class__.__name__}({self._name}) : {items}>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"

    def register(self, cls: type[_T]) -> type[_T]:
        self._set.add(cls)
        return cls

    def update(self, other: T.Self) -> T.Self:
        self._set.update(other._set)
        return self

    def union(self, other: T.Self) -> T.Self:
        return TypeRegistry().update(self).update(other)

    def __len__(self) -> int:
        return len(self._set)

    def __contains__(self, cls: type) -> bool:
        return cls in self._set or any(issubclass(cls, c) for c in self._set)

    def __iter__(self) -> T.Iterator[type]:
        return iter(self._set)


pixel_maps = TypeRegistry("pixel_maps")
