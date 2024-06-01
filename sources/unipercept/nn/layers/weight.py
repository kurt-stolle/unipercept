from __future__ import annotations

import typing as T
from functools import partial
from types import EllipsisType
from typing import Callable, Sequence, Type

import regex as re
import torch
import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

NONLINEARITY_NAMES: T.OrderedDict[
    type[nn.Module] | re.Pattern, tuple[str, float | T.Callable[[nn.Module], float]]
] = T.OrderedDict(
    {
        nn.LeakyReLU: ("leaky_relu", lambda m: m.negative_slope),
        re.compile(".*selu.*"): ("selu", 0.0),
        re.compile(".*tanh.*"): ("relu", 0.0),
        re.compile(".*sigmoid.*"): ("sigmoid", 0.0),
        re.compile(".*elu.*"): ("relu", 0.0),
    }
)


def get_nonlinearity_name(module: nn.Module) -> tuple[str, float]:
    """
    Get the name of the nonlinearity and its gain.
    """

    if isinstance(module, type):
        msg = "Cannot determine nonlinearity name and gain for module class. Pass an instance instead."
        raise ValueError(msg)

    for key, (name, gain) in NONLINEARITY_NAMES.items():
        if isinstance(key, re.Pattern):
            if key.match(module.__class__.__name__.lower()):
                break
        elif isinstance(key, type):
            if isinstance(module, key):
                break
    else:
        msg = (
            f"Nonlinearity {module.__class__.__name__} not supported. "
            "Consider registering it to NONLINEARITY_NAMES."
        )
        raise NotImplementedError(msg)

    if callable(gain):
        gain = gain(module)

    return name, gain


def init_wrap(
    modules: Sequence[Type[nn.Module]],
    *attr_init: tuple[
        EllipsisType | str,
        Callable,
        bool,
    ],
) -> Callable[T.Concatenate[nn.Module, ...], None]:
    """
    Wraps an initialization function to be applied to a module if it is an instance of one of the given modules.

    Parameters
    ----------
    modules
        Sequence of module types to apply the initialization to.
    attr_init
        Tuple of attribute name (str), initialization function (callable), and whether to forward kwargs (bool).
    """

    @torch.no_grad()
    def _init_weights(module: nn.Module, *args, **kwargs) -> None:
        if isinstance(module, tuple(modules)):
            for attr, init, use_args in attr_init:
                if attr is ...:
                    tensor_or_module = module
                    assert isinstance(tensor_or_module, nn.Module)
                else:
                    tensor_or_module = getattr(module, attr)
                    assert isinstance(tensor_or_module, torch.Tensor)
                init = T.cast(Callable[..., None], init)
                if use_args:
                    init(tensor_or_module, *args, **kwargs)
                else:
                    init(tensor_or_module, *args, **kwargs)

    return _init_weights


init_trunc_normal_ = init_wrap(
    (nn.Conv2d, nn.Linear),
    (
        "weight",
        nn.init.trunc_normal_,
        True,
    ),
    (
        "bias",
        nn.init.zeros_,
        False,
    ),
)
init_xavier_fill_ = init_wrap(
    (nn.Conv2d, nn.Linear),
    (
        ...,
        c2_xavier_fill,
        False,
    ),
)
init_msra_fill_ = init_wrap(
    (nn.Conv2d, nn.Linear),
    (
        ...,
        c2_msra_fill,
        False,
    ),
)
