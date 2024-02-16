from __future__ import annotations

import typing as T

import torch.nn as nn
import torch.utils.data

__all__ = [
    "DataLoaderFactory",
    "ModelFactory",
]

_I_co = T.TypeVar("_I_co", covariant=True)
_M_co = T.TypeVar("_M_co", bound=nn.Module, covariant=True)
_T_contra = T.TypeVar("_T_contra", contravariant=True)

DataLoaderFactory: T.TypeAlias = T.Callable[[int], torch.utils.data.DataLoader[_I_co]]
ModelFactory: T.TypeAlias = T.Callable[[_T_contra | None], _M_co]  # TODO: refactor
