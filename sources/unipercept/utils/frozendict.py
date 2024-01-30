from __future__ import annotations

from copy import deepcopy
from typing import (
    Any,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import Self, override

from unipercept.utils.descriptors import objectmagic

__all__ = ["frozendict"]

_K = TypeVar("_K", covariant=False, bound=Hashable)
_V = TypeVar("_V", covariant=True)
_X = TypeVar("_X", bound=Hashable, contravariant=True)
_Y = TypeVar("_Y", contravariant=True)
_S = TypeVar("_S", bound="frozendict")


def _frozen_method(self, *args, **kwargs):
    raise TypeError(f"'{type(self).__name__}' is frozen")


class frozendict(dict, Mapping[_K, _V], Generic[_K, _V]):
    """
    A simple immutable dictionary.

    The API is the same as `dict`, without methods that can change the
    immutability. In addition, it supports __hash__().
    """

    __slots__ = ("_hash",)

    @objectmagic
    def __or__(
        self: frozendict[_K, _V],
        other: frozendict[_X, _Y],
    ) -> frozendict[_K | _X, _V | _Y]:
        res: dict[_K | _X, _V | _Y] = {}
        res.update(dict(self.items()))
        res.update(dict(other.items()))

        return type(self)(res)

    @override
    @classmethod
    def fromkeys(cls, iterable: Iterable[_K], value: Optional[_V] = None) -> Self:
        """
        See: ``dict.fromkeys()``
        """

        return cls(super().fromkeys(iterable, value))

    @overload
    def __new__(cls, map_it=None, /) -> Self:
        ...

    @overload
    def __new__(cls, map_it: Iterable[tuple[_K, _V]] | dict[_K, _V], /) -> Self:
        ...

    @overload
    def __new__(cls, map_it: None, /, **kwargs: _V) -> frozendict[str, _V]:
        ...

    def __new__(
        cls, map_it: Iterable[tuple[_K, _V]] | dict[_K, _V] | None = None, /, **kwargs
    ):
        if map_it is not None:
            if bool(kwargs):
                raise TypeError(
                    "frozendict() takes no keyword arguments when a single positional argument is given."
                )

            if map_it.__class__ == frozendict and cls == frozendict:
                return map_it
        else:
            map_it = cast(dict[_K, _V], kwargs)
        if isinstance(map_it, dict):
            it = cast(Iterable[tuple[_K, _V]], iter(map_it.items()))
        else:
            it = map_it

        self = super().__new__(cls, it)
        super().__init__(self, it)
        return self

    def __init__(self, *args, **kwargs):
        pass  # already initialized in __new__

    @override
    def __hash__(self, *args, **kwargs) -> int:
        try:
            return self._hash
        except AttributeError:
            pass
        try:
            h = hash(frozenset(self.items()))
        except TypeError:
            h = hash(tuple(sorted(self.items())))

        object.__setattr__(self, "_hash", h)
        return h

    @override
    def __repr__(self):
        try:
            body = super().__repr__()

            return f"frozendict({body})"
        except:
            return "frozendict"

    @override
    def __str__(self):
        return repr(self)

    @override
    def copy(self) -> Self:
        r"""
        Return the object itself, as it's an immutable.
        """

        cls = type(self)

        if cls == frozendict:
            return self

        return cls(self)

    def __copy__(self) -> Self:
        r"""
        See copy().
        """

        return self.copy()

    def __deepcopy__(self, memo, *args, **kwargs) -> Self:
        r"""
        As for tuples, if hashable, see copy(); otherwise, it returns a
        deepcopy.
        """

        klass = type(self)
        return_copy = klass == frozendict

        if return_copy:
            try:
                hash(self)
            except TypeError:
                return_copy = False

        if return_copy:
            return self.copy()

        tmp = deepcopy(dict(self))

        return klass(tmp)

    @override
    def __reduce__(self, *args, **kwargs):
        return (type(self), (dict(self),))

    def set(self, key: _X, value: _Y) -> frozendict[_K | _X, _V | _Y]:
        mutable_copy = dict(*self.items())
        new_self: dict[_K | _X, _Y | _V] = deepcopy(mutable_copy)
        new_self[key] = value

        return type(self)(new_self)  # type: ignore

    @override
    def setdefault(
        self: frozendict[_K, _V],
        key: _X,
        default: _Y,
    ) -> frozendict[_K | _X, _V | _Y]:
        if key in self:
            return self  # type: ignore

        mutable_copy = dict(*self.items())
        new_self: dict[_K | _X, _Y | _V] = deepcopy(mutable_copy)
        new_self[key] = default

        return type(self)(new_self)

    def delete(self, key: Any) -> Self:
        new_self = deepcopy(dict(self))
        del new_self[key]

        if new_self:
            return type(self)(new_self)

        return type(self)()

    def _get_by_index(self, collection, index):
        try:
            return collection[index]
        except IndexError:
            maxindex = len(collection) - 1
            name = type(self).__name__
            raise IndexError(f"{name} index {index} out of range {maxindex}") from None

    def key(self, index=0) -> _K:
        collection = tuple(self.keys())

        return self._get_by_index(collection, index)

    def value(self, index=0) -> _V:
        collection = tuple(self.values())

        return self._get_by_index(collection, index)

    def item(self, index: int = 0) -> tuple[_K, _V]:
        collection = tuple(self.items())

        return self._get_by_index(collection, index)

    @override
    def __setitem__(self, key, val):
        raise TypeError(f"'{type(self).__name__}' is frozen")

    @override
    def __delitem__(self, key):
        raise TypeError(f"'{type(self).__name__}' is frozen")

    @override
    def __reversed__(self) -> Iterator[_K]:
        return reversed(tuple(self))

    clear = _frozen_method
    pop = _frozen_method
    popitem = _frozen_method
    update = _frozen_method
    __delattr__ = _frozen_method
    __setattr__ = _frozen_method
