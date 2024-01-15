import functools
from copy import copy, deepcopy
from types import MethodType
from typing import Callable, Concatenate, Generic, ParamSpec, TypeVar, overload

try:
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self  # type: ignore

_P = ParamSpec("_P")
_T = TypeVar("_T", bound=object)
_R = TypeVar("_R")

ShadowFunction = Callable[Concatenate[_T, _P], _R]


class shadowmutate(Generic[_T, _P, _R]):
    @overload
    def __new__(cls, fn: ShadowFunction, /, **kwargs) -> ShadowFunction:
        ...

    @overload
    def __new__(cls, **kwargs) -> Self:
        ...

    def __new__(cls, *args, **kwargs) -> ShadowFunction | Self:
        # Case 1: Decorator as @shadowmutate without arguments
        if len(args) == 1 and callable(args[0]):
            return cls(**kwargs)(args[0])
        elif len(args) != 0:
            raise TypeError(f"Expected 0 or 1 positional argument, got {len(args)} positional arguments")

        # Case 2: Decorator as @shadowmutate(...) with (optional) keyword arguments
        return super().__new__(cls)

    def __init__(self, *, deep: bool = False):
        self.copy = deepcopy if deep else copy

    def __call__(self, fn: ShadowFunction) -> Callable[Concatenate[_T, _P], _R]:
        """
        Decorator that calls a function with a shallow copy of the first argument.
        """

        @functools.wraps(fn)
        def wrapper(obj: _T, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            obj_shadow = self.copy(obj)
            return MethodType(fn, obj_shadow)(*args, **kwargs)

        return wrapper
