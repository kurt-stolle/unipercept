from __future__ import annotations

from functools import partial
from types import EllipsisType
from typing import Callable, Sequence, Type

import torch
import torch.nn as nn
from fvcore.nn.weight_init import c2_xavier_fill


def init_wrap(
    modules: Sequence[Type[nn.Module]],
    *attr_init: tuple[
        str,
        Callable[
            [torch.Tensor],
            torch.Tensor,
        ],
        bool,
    ]
    | tuple[
        EllipsisType,
        Callable[
            [nn.Module],
            None,
        ],
        bool,
    ],
) -> Callable[..., None]:
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
    def _init_weights(module: nn.Module, **kwargs) -> None:
        if isinstance(module, tuple(modules)):
            for attr, init, use_kwargs in attr_init:
                if attr is ...:
                    init(module, **kwargs if use_kwargs else {})
                    continue
                tensor = getattr(module, attr, None)
                if tensor is None:
                    continue
                init(tensor, **kwargs if use_kwargs else {})

    return _init_weights


init_trunc_normal_ = init_wrap(
    (nn.Conv2d, nn.Linear),
    (
        "weight",
        partial(nn.init.trunc_normal_),
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
