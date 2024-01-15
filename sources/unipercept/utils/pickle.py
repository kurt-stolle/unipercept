from __future__ import annotations

import functools
import inspect
import pickle
import typing as T
import warnings

import cloudpickle
from typing_extensions import override

__all__ = ["pickles", "as_picklable"]


_LAMBDA_NAME: T.Final = (lambda: None).__name__
_LOCALS_NAME: T.Final = "<locals>"
_ALWAYS_PICKLES: T.Final[tuple[type, ...]] = (functools.partial,)

_T = T.TypeVar("_T", bound=T.Callable)


def _has_name(obj: T.Callable, name: str) -> bool:
    """
    Check if a function has a name.
    """

    try:  # Check for __name__ attribute, if it exists
        privnaem = getattr(obj, "__name__", None)
        if privnaem is not None and name in privnaem:
            return True
    except AttributeError:
        pass  # FIXME! wrong attribute?

    try:  # Check for __qualname__ attribute, if it exists
        qualname = getattr(obj, "__qualname__", None)
        if qualname is not None and name in qualname:
            return True
    except AttributeError:
        pass  # FIXME!

    return False


def _is_safe_class(cls: T.Type) -> bool:
    """
    Check if a class is a pickling class.
    """

    if not callable(cls):
        return False
    if not inspect.isclass(cls):
        return False
    return isinstance(cls, _ALWAYS_PICKLES)


def _is_lambda(fn: T.Callable) -> bool:
    """
    Check if a function is a lambda.
    """
    return _has_name(fn, _LAMBDA_NAME)


def _is_closure(fn: T.Callable) -> bool:
    """
    Applies some heuristics on the function/class name to check if it is a local function, and thus a closure.
    """

    return _has_name(fn, _LOCALS_NAME)


def pickles(fn: T.Callable, use_experiment=True) -> bool:
    """
    Check if a function can be pickled, uses various heuristics.
    """

    if not callable(fn):
        return False

    if _is_lambda(fn) or _is_closure(fn):
        return False

    if _is_safe_class(fn):
        return True

    if use_experiment:
        try:
            pickle.dumps(fn)
            return True
        except (Exception, pickle.PicklingError) as e:
            warnings.warn(f"Cannot pickle {fn.__name__} using direct pickle experiment.", stacklevel=2)
            return False
    else:
        warnings.warn(
            "Cannot determine whether {fn.__name__} is picklable using heuristics, assuming it is not.", stacklevel=2
        )

    return False


class as_picklable(T.Generic[_T]):
    """
    A decorator that makes a function picklable, based on Detectron2's implementation ``PicklableWrapper``.
    """

    __slots__ = ("_fn",)

    def __new__(cls, fn: _T) -> _T:
        if isinstance(fn, cls):
            return T.cast(_T, fn)
        if pickles(fn):
            return fn

        fn_wrap = super().__new__(cls)
        fn_wrap._fn = fn

        return T.cast(_T, fn_wrap)

    @override
    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._fn))

    @override
    def __eq__(self, other: T.Any) -> bool:
        is_our_class = isinstance(other, self.__class__)

        if is_our_class:
            return self._fn == other._fn
        else:
            assert callable(other), f"Expected callable, got {type(other)}"
            return self._fn == other

    @override
    def __str__(self) -> str:
        return str(self._fn)

    @override
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self._fn}>"

    @override
    def __reduce__(self):
        s = cloudpickle.dumps(self._fn)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __getattr__(self, attr):
        if attr == "__slots__" or attr in self.__slots__:
            return super().__getattribute__(attr)
        else:
            return self._fn.__getattribute__(attr)

    @override
    def __setattr__(self, attr: T.Never, val: T.Never) -> T.Any:
        if attr == "__slots__" or attr in self.__slots__:
            return super().__setattr__(attr, val)
        else:
            return self._fn.__setattr__(attr, val)
