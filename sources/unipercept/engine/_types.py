from __future__ import annotations

import typing as T

import torch.nn as nn

_I_co = T.TypeVar("_I_co", covariant=True)
_M_co = T.TypeVar("_M_co", bound=nn.Module, covariant=True)
_T_contra = T.TypeVar("_T_contra", contravariant=True)
