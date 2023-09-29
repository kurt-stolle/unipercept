from __future__ import annotations

import typing as T

import torch.nn as nn
import torch.utils.data

_I_co = T.TypeVar("_I_co", covariant=True)
_M_co = T.TypeVar("_M_co", bound=nn.Module, covariant=True)
_T_contra = T.TypeVar("_T_contra", contravariant=True)


class DataLoaderFactory(T.Protocol, T.Generic[_I_co]):
    def __call__(self, batch_size: int) -> torch.utils.data.DataLoader[_I_co]:
        ...


class ModelFactory(T.Protocol, T.Generic[_T_contra, _M_co]):
    def __call__(self, trial: _T_contra | None) -> _M_co:
        ...
