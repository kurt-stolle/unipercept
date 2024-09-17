from __future__ import annotations

import typing as T
from copy import deepcopy
from itertools import repeat

from torch import nn

__all__ = [
    "to_ntuple",
    "to_1tuple",
    "to_2tuple",
    "to_3tuple",
    "to_4tuple",
    "to_5tuple",
    "to_6tuple",
    "make_divisible",
    "extend_tuple",
    "clone_to",
]


def to_ntuple[_V](n: int) -> T.Callable[[_V | T.Iterable[_V]], tuple[_V, ...]]:
    def parse(x: _V | T.Iterable[_V]) -> tuple[_V, ...]:
        if isinstance(x, T.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))  # type: ignore

    return parse


if T.TYPE_CHECKING:

    def to_1tuple[_V](x: _V | T.Iterable[_V]) -> tuple[_V]: ...
    def to_2tuple[_V](x: _V | T.Iterable[_V]) -> tuple[_V, _V]: ...
    def to_3tuple[_V](x: _V | T.Iterable[_V]) -> tuple[_V, _V, _V]: ...
    def to_4tuple[_V](x: _V | T.Iterable[_V]) -> tuple[_V, _V, _V, _V]: ...
    def to_5tuple[_V](x: _V | T.Iterable[_V]) -> tuple[_V, _V, _V, _V, _V]: ...
    def to_6tuple[_V](x: _V | T.Iterable[_V]) -> tuple[_V, _V, _V, _V, _V, _V]: ...
else:
    to_1tuple = to_ntuple(1)
    to_2tuple = to_ntuple(2)
    to_3tuple = to_ntuple(3)
    to_4tuple = to_ntuple(4)
    to_5tuple = to_ntuple(5)
    to_6tuple = to_ntuple(6)


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def extend_tuple(x, n):
    if not isinstance(x, (tuple, list)):
        x = (x,)
    else:
        x = tuple(x)
    pad_n = n - len(x)
    if pad_n <= 0:
        return x[:n]
    return x + (x[-1],) * pad_n


_CloneTargetT = T.TypeVar("_CloneTargetT", bound=nn.Module)


def clone_to(
    template: nn.Module, target: type[_CloneTargetT] = nn.ModuleList, *, n: int
) -> _CloneTargetT:
    return target([deepcopy(template) for _ in range(n)])
