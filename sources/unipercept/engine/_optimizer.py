"""Implements a lazy optimizer for use in configuration files."""

# TODO: https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.PostLocalSGDOptimizer
# TODO: https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer
from __future__ import annotations

import copy
import dataclasses as D
import enum
import functools
import itertools
import math
import typing as T
from collections import defaultdict

import regex as re
import timm.optim
import timm.optim.optim_factory
import torch.fx
import torch.optim
from torch import nn

from unipercept.log import get_logger
from unipercept.state import get_process_count

__all__ = [
    "create_optimizer",
    "OptimType",
    "OptimPackage",
    "OptimizerFactory",
    "ParameterDefinition",
    "ParameterHPs",
    "LearningRate",
]

_logger = get_logger(__name__)

Optimizer: T.TypeAlias = torch.optim.Optimizer
Params: T.TypeAlias = T.Iterable[nn.Parameter]
ModelOrParams: T.TypeAlias = nn.Module | Params

NORM_MODULE_CLASSES: T.Final = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)
NORM_MODULE_PATTERN: T.Final = re.compile(r".*Norm\d*[d]?$")
NORM_DEFAULT_WEIGHT_DECAY: T.Final = 0.0

EMBEDDING_MODULE_CLASSES: T.Final = (torch.nn.Embedding,)
EMBEDDING_MODULE_PATTERN: T.Final = re.compile(r".*Embedding\d*[d]?$")
EMBEDDING_DEFAULT_WEIGHT_DECAY: T.Final = 0.0

BIAS_DEFAULT_WEIGHT_DECAY: T.Final = None  # no change


def _check_module_type(
    module: nn.Module | torch.jit.ScriptModule | torch.fx.GraphModule,
    types: T.Iterable[type] | None = None,
    pattern: re.Pattern | None = None,
) -> bool:
    assert types is not None or pattern is not None
    if types is not None:
        if isinstance(module, tuple(types)):
            return True
    if pattern is not None:
        if isinstance(module, torch.jit.ScriptModule):
            name = module.original_name
        else:
            name = type(module).__name__
        if pattern.match(name):
            return True
    return False


def _is_norm_module(
    module: nn.Module | torch.jit.ScriptModule | torch.fx.GraphModule,
) -> bool:
    return _check_module_type(
        module, types=NORM_MODULE_CLASSES, pattern=NORM_MODULE_PATTERN
    )


def _is_embedding_module(
    module: nn.Module | torch.jit.ScriptModule | torch.fx.GraphModule,
) -> bool:
    return _check_module_type(
        module, types=EMBEDDING_MODULE_CLASSES, pattern=EMBEDDING_MODULE_PATTERN
    )


class ParameterHPs(T.TypedDict, total=False):
    """
    Hyperparameters for a single parameter in the optimizer configuration.
    """

    lr: float
    weight_decay: float


class ParameterDefinition(ParameterHPs):
    """
    Definition of a parameter in the optimizer configuration, i.e. 'params' and the hyperparameters for this parameter.
    """

    params: list[nn.Parameter]


class OptimType(enum.StrEnum):
    """
    Optimizer types supported by this module. Mostly intended to provide a clear API in configuration files.
    """

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


@D.dataclass
class LearningRate:
    """
    Container class for learning rates that are defined at instances of other parameters
    (e.g. batch size). When these parameters are changed, the learning rate is updated
    accordingly.
    """

    value: float
    batch_size: int | None = None
    per_device: bool = False

    @staticmethod
    def scale_to_batch_size(
        lr: float, opt: OptimType, cur: int | None, new: int | None
    ) -> float:
        if cur is None:
            return lr
        if new is None:
            msg = "Cannot scale learning rate to batch size without new batch size"
            raise ValueError(msg)
        if cur == new:
            return lr

        k = new / cur
        match opt:
            case (
                OptimType.ADAM,
                OptimType.ADAMW,
                OptimType.ADAMP,
                OptimType.NADAM,
                OptimType.NADAMW,
                OptimType.RADAM,
                OptimType.ADAMAX,
                OptimType.ADABELIEF,
                OptimType.RADABELIEF,
                OptimType.ADAFACTOR,
                OptimType.ADANP,
                OptimType.ADANW,
                OptimType.LAMB,
                OptimType.LAMBC,
                OptimType.LARC,
                OptimType.LARS,
                OptimType.NLARC,
                OptimType.NLARS,
                OptimType.MADGRAD,
                OptimType.MADGRADW,
                OptimType.NOVOGRAD,
                OptimType.RMSPROP,
                OptimType.RMSPROPTF,
                OptimType.LION,
                OptimType.ADAHESSIAN,
            ):
                lr *= math.sqrt(k)
            case _:
                lr *= k
        return lr

    @staticmethod
    def scale_to_processes(
        lr: float, opt: OptimType, per_device: bool, devices: int
    ) -> float:
        if not per_device:
            return lr
        return lr / devices

    def scale(
        self,
        opt: OptimType,
        *,
        batch_size: int | None,
        processes: int,
    ) -> float:
        value = self.value

        # Scale learning rate to batch size
        value = self.scale_to_batch_size(value, opt, self.batch_size, batch_size)

        # Scale learning rate to processes
        value = self.scale_to_processes(value, opt, self.per_device, processes)

        _logger.info("Scaling learning rate %.2e to %.2e", self.value, value)

        return value


class OptimPackage(enum.StrEnum):
    DEFAULT = enum.auto()
    APEX = enum.auto()
    BNB = enum.auto()
    BNB_8BIT = enum.auto()
    SCHEDULE_FREE = enum.auto()


class OptimizerFactory:
    _partial: T.Final

    def __init__(
        self,
        opt: str | OptimType | type[torch.optim.Optimizer] = OptimType.SGD,
        pkg: str | OptimPackage | None = None,
        *args,
        **kwargs,
    ):
        """
        Lazily create an optimizer, i.e. without passing the model or parameters to the optimizer constructor.

        See: ``create_optimizer``.
        """
        if isinstance(opt, str) or isinstance(opt, OptimType):
            self._partial = functools.partial(
                create_optimizer, opt, pkg or OptimPackage.DEFAULT, *args, **kwargs
            )
        elif isinstance(opt, type) and issubclass(opt, torch.optim.Optimizer):
            raise NotImplementedError("Cannot create optimizer from type")
        else:
            raise TypeError(f"Invalid optimizer type: {type(opt)}")

    def __call__(self, model_or_params: ModelOrParams, *args, **kwargs) -> Optimizer:
        return self._partial(model_or_params, *args, **kwargs)


def create_optimizer(
    opt: str | OptimType,
    pkg: str | OptimPackage,
    model_or_params: ModelOrParams,
    batch_size: int | None = None,
    /,
    *,
    lr: float | LearningRate = 5e-5,
    weight_decay: float = 0.0,
    foreach: bool | None = None,
    lookahead: bool = False,
    momentum: float = 0.9,
    weight_decay_norm: float | None = NORM_DEFAULT_WEIGHT_DECAY,
    weight_decay_embedding: float | None = EMBEDDING_DEFAULT_WEIGHT_DECAY,
    weight_decay_bias: float | None = BIAS_DEFAULT_WEIGHT_DECAY,
    lr_factor_fn: T.Callable | None = None,
    param_overrides: dict[str, ParameterHPs] | None = None,
    param_fn: T.Callable[[str, str, ParameterHPs], ParameterHPs | None] | None = None,
    extra_params: T.Iterable[ParameterDefinition] | None = None,
    **opt_args: T.Any,
) -> Optimizer:
    """
    Create an optimizer. Based on the implementation in ``timm.optim.create_optimizer``.

    Parameters
    ----------
    model_or_params:
        Model containing parameters to optimize
    opt:
        Name of optimizer to create
    lr:
        Initial learning rate
    weight_decay:
        Weight decay to apply in optimizer
    momentum:
        Momentum for momentum based optimizers (others may use betas via kwargs)
    foreach:
        Enable / disable foreach (multi-tensor) operation if True / False. Choose safe default if None
    extra_params:
        Extra parameters to pass to optimizer.
    **kwargs:
        Extra optimizer specific kwargs to pass through

    Returns
    -------
    Optimizer
        An optmizer instance
    """
    opt = OptimType(opt) if isinstance(opt, str) else opt
    pkg = OptimPackage(pkg) if isinstance(pkg, str) else pkg

    if not isinstance(lr, LearningRate):
        lr = LearningRate(lr)

    lr = lr.scale(opt, batch_size=batch_size, processes=get_process_count())

    # Extract parameters from model
    parameters = get_optimizer_params(
        model_or_params,
        lr,
        weight_decay,
        weight_decay_bias=weight_decay_bias,
        weight_decay_embedding=weight_decay_embedding,
        weight_decay_norm=weight_decay_norm,
        lr_factor_fn=lr_factor_fn,
        param_overrides=param_overrides,
        param_fn=param_fn,
        extra_params=extra_params,
    )

    # Copy the optimizer arguments to avoid modifying the original dictionary
    opt_args = dict(copy.deepcopy(opt_args))
    opt_args["lr"] = lr

    if foreach is None:
        if opt in _DEFAULT_FOREACH:
            opt_args["foreach"] = True
    else:
        opt_args["foreach"] = foreach

    opt_args["weight_decay"] = weight_decay

    # Create the optimizer
    match pkg:
        case OptimPackage.DEFAULT:
            optimizer = _create_default_optimizer(
                opt, parameters, momentum=momentum, **opt_args
            )
        case OptimPackage.APEX:
            optimizer = _create_apex_optimizer(
                opt, parameters, momentum=momentum, **opt_args
            )
        case OptimPackage.BNB:
            optimizer = _create_bnb_optimizer(
                opt, False, parameters, momentum=momentum, **opt_args
            )
        case OptimPackage.BNB_8BIT:
            optimizer = _create_bnb_optimizer(
                opt, True, parameters, momentum=momentum, **opt_args
            )
        case OptimPackage.SCHEDULE_FREE:
            optimizer = _create_schedule_free_optimizer(opt, parameters, **opt_args)
        case _:
            raise ValueError(f"Invalid optimizer package: {pkg}")

    if lookahead:
        optimizer = timm.optim.Lookahead(optimizer)

    return optimizer


_DEFAULT_FOREACH: T.Final = {OptimType.LION}


def _create_default_optimizer(
    opt: OptimType, parameters: Params, /, momentum: float, **opt_args
) -> torch.optim.Optimizer:
    """
    Use ``torch.optim`` or ``timm.optim`` to create an optimizer.
    PyTorch is the preferred choice, but Timm implements many more cutting edge optimizers.
    """
    match opt:
        case OptimType.SGD:
            opt_args.pop("eps", None)
            assert momentum is not None
            opt_args.setdefault("momentum", momentum)
            assert "nesterov" not in opt_args
            opt_args["nesterov"] = True

            optimizer = torch.optim.SGD(parameters, **opt_args)
        case OptimType.MOMENTUM:
            opt_args.pop("eps", None)
            assert momentum is not None
            opt_args.setdefault("momentum", momentum)
            assert "nesterov" not in opt_args
            opt_args["nesterov"] = False

            optimizer = torch.optim.SGD(parameters, **opt_args)
        case OptimType.SGDP:
            assert isinstance(momentum, int)
            opt_args.setdefault("momentum", momentum)
            assert "nesterov" not in opt_args
            opt_args["nesterov"] = True

            optimizer = timm.optim.SGDP(parameters, **opt_args)
        case OptimType.ADAM:
            optimizer = torch.optim.Adam(parameters, **opt_args)
        case OptimType.ADAMW:
            optimizer = torch.optim.AdamW(parameters, **opt_args)
        case OptimType.ADAMP:
            assert "nesterov" not in opt_args
            opt_args["nesterov"] = True
            assert "wd_ratio" not in opt_args
            opt_args["wd_ratio"] = 0.01
            optimizer = timm.optim.AdamP(parameters, **opt_args)
        case OptimType.NADAM:
            optimizer = timm.optim.Nadam(parameters, **opt_args)
        case OptimType.RADAM:
            optimizer = timm.optim.RAdam(parameters, **opt_args)
        case OptimType.ADAMAX:
            optimizer = torch.optim.Adamax(parameters, **opt_args)
        case OptimType.ADABELIEF:
            assert "rectify" not in opt_args
            opt_args["rectify"] = False
            optimizer = timm.optim.AdaBelief(parameters, **opt_args)
        case OptimType.RADABELIEF:
            assert "rectify" not in opt_args
            opt_args["rectify"] = True
            optimizer = timm.optim.AdaBelief(parameters, **opt_args)
        case OptimType.ADADELTA:
            optimizer = torch.optim.Adadelta(parameters, **opt_args)
        case OptimType.ADAGRAD:
            opt_args.setdefault("eps", 1e-8)
            optimizer = torch.optim.Adagrad(parameters, **opt_args)
        case OptimType.ADAFACTOR:
            optimizer = timm.optim.Adafactor(parameters, **opt_args)
        case OptimType.ADANP:
            assert "no_prox" not in opt_args
            opt_args["no_prox"] = False
            optimizer = timm.optim.Adan(parameters, **opt_args)
        case OptimType.ADANW:
            assert "no_prox" not in opt_args
            opt_args["no_prox"] = True
            optimizer = timm.optim.Adan(parameters, **opt_args)
        case OptimType.LAMB:
            assert "trust_clip" not in opt_args
            opt_args["trust_clip"] = False
            optimizer = timm.optim.Lamb(parameters, **opt_args)
        case OptimType.LAMBC:
            assert "trust_clip" not in opt_args
            opt_args["trust_clip"] = True
            optimizer = timm.optim.Lamb(parameters, **opt_args)
        case OptimType.LARC:
            assert isinstance(momentum, int)
            opt_args["momentum"] = momentum
            assert "trust_clip" not in opt_args
            opt_args["trust_clip"] = True
            optimizer = timm.optim.Lars(parameters, **opt_args)
        case OptimType.LARS:
            assert isinstance(momentum, int)
            opt_args["momentum"] = momentum
            optimizer = timm.optim.Lars(parameters, **opt_args)
        case OptimType.NLARC:
            assert isinstance(momentum, int)
            opt_args["momentum"] = momentum
            assert "trust_clip" not in opt_args
            opt_args["trust_clip"] = False
            assert "nesterov" not in opt_args
            opt_args["nesterov"] = True
            optimizer = timm.optim.Lars(
                parameters,
                **opt_args,
            )
        case OptimType.NLARS:
            assert isinstance(momentum, int)
            opt_args["momentum"] = momentum
            assert "nesterov" not in opt_args
            opt_args["nesterov"] = True
            optimizer = timm.optim.Lars(parameters, **opt_args)
        case OptimType.MADGRAD:
            assert momentum is not None
            opt_args["momentum"] = momentum
            optimizer = timm.optim.MADGRAD(parameters, **opt_args)
        case OptimType.MADGRADW:
            assert momentum is not None
            optimizer = timm.optim.MADGRAD(
                parameters, momentum=momentum, decoupled_decay=True, **opt_args
            )
        case OptimType.NOVOGRAD:
            optimizer = timm.optim.NvNovoGrad(parameters, **opt_args)
        case OptimType.RMSPROP:
            assert momentum is not None
            optimizer = torch.optim.RMSprop(
                parameters, alpha=0.9, momentum=momentum, **opt_args
            )
        case OptimType.RMSPROPTF:
            assert momentum is not None
            optimizer = timm.optim.RMSpropTF(
                parameters, alpha=0.9, momentum=momentum, **opt_args
            )
        case OptimType.LION:
            opt_args.pop("eps", None)
            optimizer = timm.optim.Lion(parameters, **opt_args)
        case OptimType.ADAHESSIAN:
            optimizer = timm.optim.Adahessian(parameters, **opt_args)
        case _:
            _error_unsupported_optimizer(opt, OptimPackage.DEFAULT)

    return optimizer


def _create_apex_optimizer(
    opt: OptimType, parameters: Params, /, momentum: float, **opt_args
) -> torch.optim.Optimizer:
    """
    Create an optimizer using the Apex package.

    The Apex package provides a collection of PyTorch optimizers that can be used to speed up training and reduce memory
    usage. These optimizers are designed to work well on modern hardware, such as GPUs and TPUs, and can be used to
    train  large-scale deep learning models.

    If the `opt` argument is set to OptimType.SGD, the FusedSGD optimizer will be used. This optimizer uses a fused
    kernel to perform the SGD update, which can lead to faster training times and reduced memory usage. However, the
    FusedSGD optimizer may not be compatible with all models or hardware configurations, so it is important to test
    carefully when using this optimizer.

    Parameters
    ----------
    opt : OptimType
        The type of optimizer to create.
    parameters : Params
        The model parameters to optimize.
    momentum : float or None, optional
        The momentum value for the optimizer (if applicable).
    opt_args : dict
        Additional optimizer arguments.

    Returns
    -------
    torch.optim.Optimizer
        The created optimizer.

    Raises
    ------
    RuntimeError
        If CUDA is not available.
    ValueError
        If an invalid optimizer type is specified.
    """
    import apex.optimizers  # pyright: ignore[reportMissingImports]

    if not torch.cuda.is_available():
        raise RuntimeError("apex optimizers require CUDA to be available.")

    match opt:
        case OptimType.SGD:
            assert momentum is not None
            opt_args.pop("eps", None)
            optimizer = apex.optimizers.FusedSGD(
                parameters, momentum=momentum, nesterov=True, **opt_args
            )
        case OptimType.MOMENTUM:
            assert momentum is not None
            opt_args.pop("eps", None)
            optimizer = apex.optimizers.FusedSGD(
                parameters, momentum=momentum, nesterov=False, **opt_args
            )
        case OptimType.ADAM:
            optimizer = apex.optimizers.FusedAdam(
                parameters, adam_w_mode=False, **opt_args
            )
        case OptimType.ADAMW:
            optimizer = apex.optimizers.FusedAdam(
                parameters, adam_w_mode=True, **opt_args
            )
        case OptimType.LAMB:
            optimizer = apex.optimizers.FusedLAMB(parameters, **opt_args)
        case OptimType.NOVOGRAD:
            opt_args.setdefault("betas", (0.95, 0.98))
            optimizer = apex.optimizers.FusedNovoGrad(parameters, **opt_args)
        case _:
            _error_unsupported_optimizer(opt, OptimPackage.APEX)
    return optimizer


def _create_bnb_optimizer(
    opt: OptimType, eight: bool, parameters: Params, /, momentum: float, **opt_args
) -> torch.optim.Optimizer:
    """
    Create an optimizer using the bitsandbytes package.

    The bitsandbytes package provides a collection of optimized PyTorch optimizers that can be used to speed up training
    and reduce memory usage. These optimizers are designed to work well on modern hardware, such as GPUs and TPUs, and
    can be used to train large-scale deep learning models.

    If the `eight` argument is set to True, the 8-bit variant of the optimizer will be used. These optimizers use 8-bit
    gradients and weights to reduce memory usage and increase training speed. However, using 8-bit precision can also
    lead to numerical instability and reduced model accuracy, so it is important to carefully tune the optimizer
    hyperparameters when using the 8-bit variants.

    Parameters
    ----------
    opt : OptimType
        The type of optimizer to create.
    eight : bool
        Whether to use the 8-bit variant of the optimizer.
    parameters : Params
        The model parameters to optimize.
    momentum : float or None, optional
        The momentum value for the optimizer (if applicable).
    opt_args : dict
        Additional optimizer arguments.

    Returns
    -------
    torch.optim.Optimizer
        The created optimizer.

    Raises
    ------
    RuntimeError
        If CUDA is not available.
    ValueError
        If an invalid optimizer type is specified.
    """
    import bitsandbytes as bnb  # pyright: ignore[reportMissingImports]

    if not torch.cuda.is_available():
        raise RuntimeError("bitsandbytes optimizers require CUDA to be available.")

    match opt:
        case OptimType.SGD:
            cls = bnb.optim.SGD if not eight else bnb.optim.SGD8bit
            opt_args.pop("eps", None)
            optimizer = cls(parameters, momentum=momentum, nesterov=True, **opt_args)
        case OptimType.MOMENTUM:
            cls = bnb.optim.SGD if not eight else bnb.optim.SGD8bit
            opt_args.pop("eps", None)
            optimizer = cls(parameters, momentum=momentum, **opt_args)
        case OptimType.ADAM:
            cls = bnb.optim.Adam if not eight else bnb.optim.Adam8bit
            optimizer = cls(parameters, **opt_args)
        case OptimType.ADAMW:
            cls = bnb.optim.AdamW if not eight else bnb.optim.AdamW8bit
            optimizer = cls(parameters, **opt_args)
        case OptimType.LAMB:
            cls = bnb.optim.Lamb if not eight else bnb.optim.Lamb8bit
            optimizer = bnb.optim.LAMB(parameters, **opt_args)
        case OptimType.LARS:
            cls = bnb.optim.Lars if not eight else bnb.optim.Lars8bit
            optimizer = cls(parameters, **opt_args)
        case OptimType.LION:
            cls = bnb.optim.Lion if not eight else bnb.optim.Lion8bit
            optimizer = cls(parameters, **opt_args)
        case _:
            _error_unsupported_optimizer(opt, OptimPackage.BNB)
    return optimizer


def _create_schedule_free_optimizer(
    opt: OptimType, parameters: Params, /, **opt_args
) -> torch.optim.Optimizer:
    """
    Create an optimizer using the the schedulefree package

    See Also
    --------
    - `GitHub <https://github.com/facebookresearch/schedule_free>`_ repository

    """
    import schedulefree  # pyright: ignore[reportMissingImports]

    match opt:
        case OptimType.ADAMW:
            optimizer = schedulefree.AdamWScheduleFree(parameters, **opt_args)
        case OptimType.SGD:
            optimizer = schedulefree.SGDScheduleFree(parameters, **opt_args)
        case _:
            _error_unsupported_optimizer(opt, OptimPackage.SCHEDULE_FREE)
    return optimizer


def _error_unsupported_optimizer(opt: OptimType, pkg: OptimPackage) -> T.NoReturn:
    msg = f"Optimizer {str(opt)!r} not supported in package {str(pkg)!r}."
    raise NotImplementedError(msg)


def get_optimizer_params(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    *,
    weight_decay_norm: float | None = NORM_DEFAULT_WEIGHT_DECAY,
    weight_decay_embedding: float | None = EMBEDDING_DEFAULT_WEIGHT_DECAY,
    weight_decay_bias: float | None = BIAS_DEFAULT_WEIGHT_DECAY,
    lr_factor_fn: T.Callable | None = None,
    param_overrides: dict[str, ParameterHPs] | None = None,
    param_fn: T.Callable[[str, str, ParameterHPs], ParameterHPs | None] | None = None,
    extra_params: T.Iterable[ParameterDefinition] | None = None,
) -> list[ParameterDefinition]:
    if param_overrides is None:
        param_overrides = {}
    defaults: ParameterHPs = {"lr": lr, "weight_decay": weight_decay}
    bias_overrides: ParameterHPs = {}
    if weight_decay_bias is not None:
        bias_overrides["weight_decay"] = weight_decay_bias
    if len(bias_overrides):
        if "bias" in param_overrides:
            raise ValueError("Conflicting overrides for 'bias'")
        param_overrides["bias"] = bias_overrides
    if lr_factor_fn is not None:
        if lr is None:
            raise ValueError("lr_factor_func requires base_lr")
    params: list[ParameterDefinition] = []
    memo: set[torch.nn.parameter.Parameter] = set()
    for module_name, module in model.named_modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            qualified_name = f"{module_name}.{module_param_name}"
            if not value.requires_grad:
                _logger.debug("Skipping parameter (frozen): %s", qualified_name)
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            hyperparams = copy.copy(defaults)
            if callable(lr_factor_fn):
                hyperparams["lr"] *= lr_factor_fn(module_name, module_param_name)

            hyperparams.update(param_overrides.get(module_param_name, {}))

            if _is_norm_module(module) and weight_decay_norm is not None:
                hyperparams["weight_decay"] = weight_decay_norm
            if _is_embedding_module(module) and weight_decay_embedding is not None:
                hyperparams["weight_decay"] = weight_decay_embedding
            if module_param_name == "param":
                hyperparams["weight_decay"] = 0.0
            if callable(param_fn):
                param_specifc_overrides = param_fn(
                    module_name, module_param_name, hyperparams
                )
                if param_specifc_overrides is None:
                    value.requires_grad_(False)
                    _logger.debug("Skipping parameter (filtered): %s", qualified_name)
                    continue  # skip this parameter
                if len(param_specifc_overrides) > 0:
                    _logger.debug(
                        "Overriding parameter: '%s' <- %s",
                        qualified_name,
                        str(param_specifc_overrides),
                    )
                    hyperparams.update(param_specifc_overrides)

            params.append({"params": [value], **hyperparams})
    defs = _simplify_groups(params)
    if extra_params is not None:
        defs.extend(extra_params)
    return defs


def _expand_param_groups(
    params: list[ParameterDefinition],
) -> list[ParameterDefinition]:
    # Transform parameter groups into per-parameter structure.
    # Later items in `params` can overwrite parameters set in previous items.
    ret = defaultdict(dict)
    for item in params:
        assert "params" in item
        cur_params = {
            x: y for x, y in item.items() if x != "params" and x != "param_names"
        }
        if "param_names" in item:
            for param_name, param in zip(
                item["param_names"], item["params"], strict=False
            ):
                ret[param].update(
                    {"param_names": [param_name], "params": [param], **cur_params}
                )
        else:
            for param in item["params"]:
                ret[param].update({"params": [param], **cur_params})
    return list(ret.values())


def _simplify_groups(params: list[ParameterDefinition]) -> list[ParameterDefinition]:
    params = _expand_param_groups(params)
    groups = defaultdict(list)  # re-group all parameter groups by their hyperparams
    for item in params:
        cur_params = tuple(
            (x, y) for x, y in item.items() if x != "params" and x != "param_names"
        )
        groups[cur_params].append({"params": item["params"]})
        if "param_names" in item:
            groups[cur_params][-1]["param_names"] = item["param_names"]

    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur["params"] = list(
            itertools.chain.from_iterable([params["params"] for params in param_values])
        )
        if len(param_values) > 0 and "param_names" in param_values[0]:
            cur["param_names"] = list(
                itertools.chain.from_iterable(
                    [params["param_names"] for params in param_values]
                )
            )
        ret.append(cur)
    return ret
