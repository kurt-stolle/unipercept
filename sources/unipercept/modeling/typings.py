"""
Implements various static types for working with neural network modules in PyTorch.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from tensordict import TensorDict

__all__ = [
    "HW",
    "Activation",
    "Norm",
    "Module",
    "ChannelModule",
    "DictModule",
    "ListModule",
    "TensorModule",
    "ConvModule",
]

_O = T.TypeVar(
    "_O",
    bound=torch.Tensor | T.Dict[str, torch.Tensor] | TensorDict | T.List[torch.Tensor] | T.Tuple[torch.Tensor],
    # covariant=True,
)

HW: T.TypeAlias = T.Tuple[int, int]
Activation: T.TypeAlias = T.Callable[..., nn.Module]
Norm: T.TypeAlias = T.Callable[[int], nn.Module]


class Module(T.Generic[_O], T.Protocol):
    def forward(self, inputs: _O) -> _O:
        ...

    def __call__(self, inputs: _O) -> _O:
        ...


class ChannelModule(Module[_O], T.Protocol):
    in_channels: T.Final[int]
    out_channels: T.Final[int]


class DictModule(Module[T.Dict[str, torch.Tensor] | TensorDict], T.Protocol):
    ...


class ListModule(Module[T.List[torch.Tensor]], T.Protocol):
    ...


class TensorModule(Module[torch.Tensor], T.Protocol):
    ...


class ConvModule(ChannelModule[torch.Tensor], T.Protocol):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        bias: bool,
        **kwargs,
    ):
        ...

    @classmethod
    def with_norm_activation(
        cls,
        *args,
        norm: T.Optional[Norm] = None,
        activation: T.Optional[Activation] = None,
        **kwargs,
    ) -> T.Self:
        ...

    @classmethod
    def with_norm(cls, *args, norm: T.Optional[Norm] = None, **kwargs) -> T.Self:
        ...

    @classmethod
    def with_activation(cls, *args, activation: T.Optional[Activation] = None, **kwargs) -> T.Self:
        ...
