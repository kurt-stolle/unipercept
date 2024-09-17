from __future__ import annotations

import typing as T

import torch
from torch import nn

EPS_FLOAT: T.Final[float] = torch.finfo(torch.float).resolution
EPS_HALF: T.Final[float] = torch.finfo(torch.half).resolution
EPS_BF16: T.Final[float] = torch.finfo(torch.bfloat16).resolution


class ScaledLossMixin(nn.Module):
    __constants__ = ("scale",)

    def __init__(self, *, scale: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.scale = scale


class StableLossMixin(nn.Module):
    __constants__ = ("eps",)

    def __init__(self, *, eps: float = EPS_FLOAT, **kwargs):
        super().__init__(**kwargs)

        self.eps = eps
