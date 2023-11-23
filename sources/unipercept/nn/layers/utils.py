from typing import Type

from torch import nn

from itertools import repeat
import collections.abc
from unipercept.nn.typings import Activation, Norm


def wrap_norm(name: str) -> Norm:
    from detectron2.layers import get_norm

    return lambda out_channels: get_norm(name, out_channels)


def wrap_activation(module: Type[nn.Module], **kwargs) -> Activation:
    def _new(**kwargs_ins):
        for key, value in kwargs_ins.items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(
                    f"Conflicting values for `{key}` passed in wrapped and instantiation call: {kwargs[key]} vs {value}"
                )

        kwargs_ins.update(kwargs)

        return module(**kwargs_ins)

    return _new


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
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