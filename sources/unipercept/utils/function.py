from __future__ import annotations

import collections.abc
import inspect
from functools import partial, wraps
from itertools import repeat
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
    overload,
)

import torch
from torch._dynamo import allow_in_graph
from typing_extensions import TypeVarTuple

__all__ = []

_Ps = ParamSpec("_Ps")
_RM = TypeVarTuple("_RM")

_R1 = TypeVar("_R1")
_R2 = TypeVar("_R2")
_R3 = TypeVar("_R3")
_R4 = TypeVar("_R4")
_R5 = TypeVar("_R5")
_R6 = TypeVar("_R6")
_R7 = TypeVar("_R7")
_R8 = TypeVar("_R8")


# Currently, the Python typing system doe not support casting each of the elements of a tuple to a different type,
# in our case a list, so we hardcode the first 9 cases.
@overload
def multi_apply(
    func: Callable[_Ps, tuple[_R1, _R2, _R3, _R4, _R5, _R6, _R7, _R8]],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[
    list[_R1],
    list[_R2],
    list[_R3],
    list[_R4],
    list[_R5],
    list[_R6],
    list[_R7],
    list[_R8],
]:
    ...


@overload
def multi_apply(
    func: Callable[_Ps, tuple[_R1, _R2, _R3, _R4, _R5, _R6, _R7]],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[list[_R1], list[_R2], list[_R3], list[_R4], list[_R5], list[_R6], list[_R7]]:
    ...


@overload
def multi_apply(
    func: Callable[_Ps, tuple[_R1, _R2, _R3, _R4, _R5, _R6]],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[list[_R1], list[_R2], list[_R3], list[_R4], list[_R5], list[_R6]]:
    ...


@overload
def multi_apply(
    func: Callable[_Ps, tuple[_R1, _R2, _R3, _R4, _R5]],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[list[_R1], list[_R2], list[_R3], list[_R4], list[_R5]]:
    ...


@overload
def multi_apply(
    func: Callable[_Ps, tuple[_R1, _R2, _R3, _R4]],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[list[_R1], list[_R2], list[_R3], list[_R4]]:
    ...


@overload
def multi_apply(
    func: Callable[_Ps, tuple[_R1, _R2, _R3]],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[list[_R1], list[_R2], list[_R3]]:
    ...


@overload
def multi_apply(
    func: Callable[_Ps, tuple[_R1, _R2]],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[list[_R1], list[_R2]]:
    ...


@overload
def multi_apply(
    func: Callable[_Ps, _R1],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[list[_R1]]:
    ...


# @overload
# def multi_apply(
#     func: Callable[_Ps, tuple[Unpack[_RM]]],
#     *args: Iterable[Any],
#     **kwargs: Any,
# ) -> tuple[Unpack[_RM]]:  # technically not correct; each item will be cast to a list.
#     ...


def multi_apply(
    func: Callable[_Ps, Any],
    *args: Iterable[Any],
    **kwargs: Any,
) -> tuple[list[Any], ...]:
    """
    Maps each argument iterable to a function, then transposes the results.
    Keyword arguments can be provided as parameters to all functions.

    Parameters
    ----------
    func : Callable.
        Any callable.
    *args: Iterable[...]
        Mapped parameters.
    **kwargs:
        Constant parameters.

    Returns
    -------
    Transposed mapping
    """

    pfunc = partial(func, **kwargs) if kwargs else func

    res_lon = map(pfunc, *args)
    res_zip = zip(*res_lon)
    return tuple(map(list, res_zip))  # type: ignore


_Pt = ParamSpec("_Pt")
_Rt = TypeVar("_Rt", covariant=True)


def traceable(f: Callable[_Pt, _Rt]) -> Callable[_Pt, _Rt]:
    f = cast(Callable[_Pt, _Rt], allow_in_graph(f))

    @wraps(f)
    def wrapper(*args: _Pt.args, **kwargs: _Pt.kwargs):
        return f(*args, **kwargs)

    return wrapper


_T = TypeVar("_T", int, float, str, torch.Tensor, bool)


@overload
def to_ntuple(n: Literal[1]) -> Callable[[_T | Iterable[_T]], tuple[_T]]:
    ...


@overload
def to_ntuple(n: Literal[2]) -> Callable[[_T | Iterable[_T]], tuple[_T, _T]]:
    ...


@overload
def to_ntuple(n: Literal[3]) -> Callable[[_T | Iterable[_T]], tuple[_T, _T, _T]]:
    ...


@overload
def to_ntuple(n: Literal[4]) -> Callable[[_T | Iterable[_T]], tuple[_T, _T, _T, _T]]:
    ...


def to_ntuple(n: int) -> Callable[[_T | Iterable[_T]], tuple[_T, ...]]:
    """Ensure that the input, which is optionally an iterable, is casted to a tuple of length `n`."""

    def parse(x: _T | Iterable[_T]) -> tuple[_T, ...]:
        res: tuple[_T, ...]
        if (
            isinstance(x, collections.abc.Iterable) or inspect.isgenerator(x)
        ) and not isinstance(x, str):
            res = tuple(v for v in x)[:n]
        else:
            res = tuple(repeat(x, n))  # type: ignore

        assert isinstance(res, tuple), f"Expect tuple, but got {type(res)}"
        assert len(res) == n, f"Expect tuple of length {n}, but got {res}"

        return res

    return parse


to_1tuple = to_ntuple(1)
to_2tuple = to_ntuple(2)
to_3tuple = to_ntuple(3)
to_4tuple = to_ntuple(4)
to_ntuple = to_ntuple


def to_sequence(x: _T | Sequence[_T]) -> Sequence[_T]:
    """Ensure that the input is casted to a tuple when it is not yet a sequence."""
    if isinstance(x, collections.abc.Sequence) and not isinstance(x, str):
        return x
    return (x,)  # type: ignore


_Td = TypeVar("_Td", float, torch.Tensor)


def make_divisible(
    v: _Td, divisor: int = 8, min_value: int | None = None, round_limit: _Td = 0.9
) -> _Td:
    """Ensure that the input is divisible by `divisor`."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v  # type: ignore


def extend_tuple(x: tuple[_T] | list[_T] | _T, n: int) -> tuple[_T, ...]:
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n
