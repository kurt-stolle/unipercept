"""
Various descriptors for dataset info (metadata about the dataset)
"""
from __future__ import annotations

from types import resolve_bases
from typing import Any, Callable, Final, Generic, Iterable, TypeAlias, TypeVar

from typing_extensions import final, override

__all__ = ["infomethod", "infoproperty", "discover_info"]

_T = TypeVar("_T", bound=Any)  # generic (any)
_R = TypeVar("_R", covariant=True, bound=Any)  # return value
_F: TypeAlias = Callable[[], _R]  # function
_F_co = TypeVar("_F_co", covariant=True, bound=Callable)


@final
class infoproperty(property):
    """
    Declare a data info property, i.e. a property that returns an entry in the info dict.
    """

    pass


@final
class infomethod(Generic[_T, _R]):
    """
    Declare a data info function, i.e. a property that returns a (partial) info dict.
    """

    def __init__(self, fn: _F[_R]):
        self._fn: Final = fn
        self._name = fn.__name__

    @property
    def name(self):
        return self._name

    @property
    @override
    def __doc__(self):
        return self._fn.__doc__

    def __set_name__(self, owner: type, name: str):
        self._name = name

    def __get__(self, ins: object | None, owner: type) -> _F[_R]:
        return self._fn

    def __call__(self, *args, **kwds) -> _R:
        return self._fn(*args, **kwds)


INFORMATIVE_DESCRIPTORS = (infoproperty, infomethod)


def discover_info(__cls: Any) -> Iterable[Any]:
    if len(__cls.__dict__) == 0:
        yield from ()
        return

    for obj in __cls.__dict__.items():
        if callable(obj) and isinstance(obj, INFORMATIVE_DESCRIPTORS):
            yield obj
