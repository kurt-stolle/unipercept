from __future__ import annotations

from functools import update_wrapper
from types import MethodType
from typing import (
    Any,
    Callable,
    Concatenate,
    Final,
    Generic,
    NoReturn,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
    overload,
)

from typing_extensions import override

from unipercept.utils.missing import MissingValue

__all__ = ["objectmagic", "immutable"]

_P = ParamSpec("_P")
_T = TypeVar("_T", bound=object)
_R = TypeVar("_R", covariant=True)


class objectmagic(Generic[_T, _P, _R]):
    """
    A variant on classmethod that can only be called on instances of a class.
    Useful for wrapping magic methods while still preserving the ability to call such a method statically.
    """

    @property
    def __func__(self) -> Callable[Concatenate[_T, _P], _R]:
        ...

    def __init__(self, fn: Callable[Concatenate[_T, _P], _R]) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.owner = None

        update_wrapper(self, fn)  # type: ignore

    def __set_name__(self, owner: type[_T], name: str) -> None:
        self.name = name
        self.owner = owner

    @overload
    def __get__(self, obj: None, *args, **kwargs) -> Callable[_P, _R]:
        ...

    @overload
    def __get__(self, obj: _T, *args, **kwargs) -> Callable[Concatenate[_T, _P], _R]:
        ...

    def __get__(
        self, obj: _T | None, *args, **kwargs
    ) -> Callable[Concatenate[_T, _P], _R] | Callable[_P, _R]:
        if obj is None:
            return self.fn  # (obj_a, obj_b)
        else:
            method = MethodType(self.fn, obj)  # type: ignore
            return cast(Callable[_P, _R], method)  # (obj_b)


_V = TypeVar("_V", covariant=True)
_NA: Final = MissingValue("NA")


class immutable(Generic[_V]):
    """
    A value that can only be set once and not changed afterwards.
    """

    def __init__(self, value: _V | MissingValue = _NA):
        self.value: _V = cast(_V, value)
        self.name: str | None = None

    def __get__(self, obj: object, cls: type[object]) -> _V:
        res = self.value
        if _NA.is_value(res):
            return res
        else:
            raise AttributeError(f"{self.name} is not set")

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.name = name

    def __set__(self, obj: object, value: Any, /) -> NoReturn | None:
        if self.name is not None:
            raise AttributeError(f"{self.name} is read-only")
        else:
            self.value = value

    @override
    def __str__(self):
        return str(self.value)

    @override
    def __repr__(self):
        return f"<immutable {type(self.value)} {str(self)}>"


class private(Generic[_V]):
    """
    A property that can only be accessed by the class that owns it. It is not accessible by subclasses.
    This is useful for properties that are only meant to be applicable by the class itself, like a version string.

    Note that this is not a security feature, and is not meant to be used as such.
    """

    value: _V | MissingValue
    owner: type | None
    name: str | None

    def __init__(
        self, value: _V | MissingValue = _NA, owner: type[object] | None = None
    ):
        self.value = value
        self.owner: type | None = owner
        self.name: str | None = repr(self)

    @override
    def __str__(self) -> str:
        res = []
        if self.owner is not None:
            res.append(f"{self.owner.__name__}")
        if self.name is not None:
            res.append(self.name)
        else:
            res.append("(unbound)")
        rep = ".".join(res)
        if self.value is not _NA:
            rep += "=" + repr(rep)
        return rep

    @override
    def __repr__(self) -> str:
        return f"<private property {str(self)}>"

    def __get__(self, obj: object, cls: type[object]):
        if self.value in _NA:
            raise _NA.Error()
        else:
            return self.value

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.name = name

    def __set__(self, obj: object, value: Any, /) -> NoReturn | None:
        if self.owner is not None:
            raise AttributeError(f"{self.name} property is read-only")
        else:
            self.value = value

    # __class_getitem__ = classmethod(types.GenericAlias)


class blocked:
    def __init__(self, fn: Optional[Callable[..., Any]] = None):
        self.name = fn.__name__ if fn is not None else "method"
        self.owner = None

    def __set_name__(self, owner: type[object], name: str) -> None:
        self.name = name
        self.owner = owner

    def __get__(self, obj: object, cls: type[object]) -> Callable[..., Any]:
        if obj is None:
            return MethodType(self, cls)
        else:
            return MethodType(self, obj)

    def __call__(self, obj, *args, **kwargs):
        raise AttributeError(f"{self.name} is blocked on {obj}")
