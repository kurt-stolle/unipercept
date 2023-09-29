from __future__ import annotations

import enum
import typing as T

import torch.nn as nn
import torch.optim

_O = T.TypeVar("_O", bound=torch.optim.Optimizer)
_P = T.ParamSpec("_P")

Optimizer: T.TypeAlias = torch.optim.Optimizer
Params: T.TypeAlias = T.Iterable[nn.Parameter]
ModelOrParams: T.TypeAlias = nn.Module | Params

__dir__ = ["create_optimizer", "OptimType", "OptimPackage", "OptimizerFactory"]

class OptimType(enum.StrEnum):
    SGD = enum.auto()
    MOMENTUM = enum.auto()
    SGDP = enum.auto()
    ADAM = enum.auto()
    ADAMW = enum.auto()
    ADAMP = enum.auto()
    NADAM = enum.auto()
    NADAMW = enum.auto()
    RADAM = enum.auto()
    ADAMAX = enum.auto()
    ADABELIEF = enum.auto()
    RADABELIEF = enum.auto()
    ADADELTA = enum.auto()
    ADAGRAD = enum.auto()
    ADAFACTOR = enum.auto()
    ADANP = enum.auto()
    ADANW = enum.auto()
    LAMB = enum.auto()
    LAMBC = enum.auto()
    LARC = enum.auto()
    LARS = enum.auto()
    NLARC = enum.auto()
    NLARS = enum.auto()
    MADGRAD = enum.auto()
    MADGRADW = enum.auto()
    NOVOGRAD = enum.auto()
    RMSPROP = enum.auto()
    RMSPROPTF = enum.auto()
    LION = enum.auto()
    ADAHESSIAN = enum.auto()

class OptimPackage(enum.Enum):
    DEFAULT = enum.auto()
    APEX = enum.auto()
    BNB = enum.auto()
    BNB_8BIT = enum.auto()

class OptimizerFactory:
    @T.overload
    def __init__(
        self,
        opt: str | OptimType,
        pkg: str | OptimPackage | None = OptimPackage.DEFAULT,
        *,
        lr: float | None = None,
        foreach: bool | None = None,
        lookahead: bool = False,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        filter_bias_and_bn=True,
        layer_decay: float | None = None,
        param_group_fn: T.Callable | None = None,
        **opt_args: T.Any,
    ): ...
    @T.overload
    def __init__(
        self,
        opt: T.Callable[T.Concatenate[T.Any, _P], _O],
        pkg: None,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ): ...
    def __init__(
        self,
        opt: str | OptimType | type[torch.optim.Optimizer] = OptimType.SGD,
        pkg: str | OptimPackage | None = None,
        *,
        lr: float | None = None,
        foreach: bool | None = None,
        lookahead: bool = False,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        filter_bias_and_bn=True,
        layer_decay: float | None = None,
        param_group_fn: T.Callable | None = None,
        **opt_args: T.Any,
    ): ...
    def __call__(self, model_or_params: ModelOrParams) -> Optimizer: ...

def create_optimizer(
    opt: str | OptimType,
    pkg: str | OptimPackage,
    model_or_params: torch.nn.Module | T.Iterable[torch.nn.Parameter],
    *,
    lr: float | None = None,
    foreach: bool | None = None,
    lookahead: bool = False,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    filter_bias_and_bn=True,
    layer_decay: float | None = None,
    param_group_fn: T.Callable | None = None,
    **opt_args: T.Any,
) -> torch.optim.Optimizer: ...
