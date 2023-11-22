"""Implements a lazy optimizer for use in configuration files."""

from __future__ import annotations

import copy
import enum
import functools
import logging
import typing as T

import timm.optim
import timm.optim.optim_factory
import torch.nn as nn
import torch.optim

from accelerate import PartialState

__all__ = ["create_optimizer", "OptimType", "OptimPackage", "OptimizerFactory"]

Optimizer: T.TypeAlias = torch.optim.Optimizer
Params: T.TypeAlias = T.Iterable[nn.Parameter]
ModelOrParams: T.TypeAlias = nn.Module | Params


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


class OptimPackage(enum.StrEnum):
    DEFAULT = enum.auto()
    APEX = enum.auto()
    BNB = enum.auto()
    BNB_8BIT = enum.auto()


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
            self._partial = functools.partial(create_optimizer, opt, pkg or OptimPackage.DEFAULT, *args, **kwargs)
        elif isinstance(opt, type) and issubclass(opt, torch.optim.Optimizer):
            self._partial = functools.partial(_wrap_lazy, opt, *args, **kwargs)
        else:
            raise TypeError(f"Invalid optimizer type: {type(opt)}")

    def __call__(self, model_or_params: ModelOrParams) -> Optimizer:
        return self._partial(model_or_params)


def _wrap_lazy(opt, model_or_params, *args, **kwargs):
    params, weight_decay = _list_optimizer_params(
        model_or_params,
        kwargs.pop("weight_decay", 0.0),
        filter_bias_and_bn=kwargs.pop("filter_bias_and_bn", True),
        layer_decay=kwargs.pop("layer_decay", None),
        param_group_fn=kwargs.pop("param_group_fn", None),
    )

    kwargs["weight_decay"] = weight_decay
    return opt(params, *args, **kwargs)


def create_optimizer(
    opt: str | OptimType,
    pkg: str | OptimPackage,
    model_or_params: ModelOrParams,
    /,
    *,
    lr: float | None = 5e-5,
    foreach: bool | None = None,
    lookahead: bool = False,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    filter_bias_and_bn=True,
    layer_decay: float | None = None,
    param_group_fn: T.Callable | None = None,
    **opt_args: T.Any,
) -> Optimizer:
    """
    Create an optimizer. Based on the implementation in ``timm.optim.create_optimizer``, with some edits to make it
    fit in our framework.

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
    filter_bias_and_bn:
        Filter out bias, bn and other 1d params from weight decay
    **kwargs:
        Extra optimizer specific kwargs to pass through

    Returns
    -------
    Optimizer
        An optmizer instance
    """
    if not (isinstance(model_or_params, nn.Module) or isinstance(model_or_params, T.Iterable)):
        raise TypeError(f"Invalid model/param type: {type(model_or_params)}")

    opt = OptimType(opt) if isinstance(opt, str) else opt
    pkg = OptimPackage(pkg) if isinstance(pkg, str) else pkg

    # Extract parameters from model
    parameters, weight_decay = _list_optimizer_params(
        model_or_params,
        weight_decay,
        filter_bias_and_bn=filter_bias_and_bn,
        layer_decay=layer_decay,
        param_group_fn=param_group_fn,
    )

    # Copy the optimizer arguments to avoid modifying the original dictionary
    opt_args = dict(copy.deepcopy(opt_args))
    if lr is not None:
        opt_args["lr"] = lr * PartialState().num_processes
    if foreach is None:
        if opt in _DEFAULT_FOREACH:
            opt_args["foreach"] = True
    else:
        opt_args["foreach"] = foreach
    opt_args["weight_decay"] = weight_decay

    # Create the optimizer
    match pkg:
        case OptimPackage.DEFAULT:
            optimizer = _create_default_optimizer(opt, parameters, momentum=momentum, **opt_args)
        case OptimPackage.APEX:
            optimizer = _create_apex_optimizer(opt, parameters, momentum=momentum, **opt_args)
        case OptimPackage.BNB:
            optimizer = _create_bnb_optimizer(opt, False, parameters, momentum=momentum, **opt_args)
        case OptimPackage.BNB_8BIT:
            optimizer = _create_bnb_optimizer(opt, True, parameters, momentum=momentum, **opt_args)
        case _:
            raise ValueError(f"Invalid optimizer package: {pkg}")

    if lookahead:
        optimizer = timm.optim.Lookahead(optimizer)

    return optimizer


_DEFAULT_FOREACH: T.Final = {OptimType.LION}


def _create_default_optimizer(
    opt: OptimType, parameters: Params, /, momentum: float, **opt_args
) -> torch.optim.Optimizer:
    """Use ``torch.optim`` or ``timm.optim`` to create an optimizer. PyTorch is the preferred choice, but Timm implements many more cutting edge optimizers."""
    match opt:
        case OptimType.SGD:
            assert momentum is not None
            opt_args.pop("eps", None)
            optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=True, **opt_args)
        case OptimType.MOMENTUM:
            assert momentum is not None
            opt_args.pop("eps", None)
            optimizer = torch.optim.SGD(parameters, momentum=momentum, nesterov=False, **opt_args)
        case OptimType.SGDP:
            assert momentum is not None
            optimizer = timm.optim.SGDP(parameters, momentum=momentum, nesterov=True, **opt_args)
        case OptimType.ADAM:
            optimizer = torch.optim.Adam(parameters, **opt_args)
        case OptimType.ADAMW:
            optimizer = torch.optim.AdamW(parameters, **opt_args)
        case OptimType.ADAMP:
            optimizer = timm.optim.AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
        case OptimType.NADAM:
            optimizer = timm.optim.Nadam(parameters, **opt_args)
        # case OptimType.NADAMW:
        #     optimizer = timm.optim.NAdamW(parameters, **opt_args)
        case OptimType.RADAM:
            optimizer = timm.optim.RAdam(parameters, **opt_args)
        case OptimType.ADAMAX:
            optimizer = torch.optim.Adamax(parameters, **opt_args)
        case OptimType.ADABELIEF:
            optimizer = timm.optim.AdaBelief(parameters, rectify=False, **opt_args)
        case OptimType.RADABELIEF:
            optimizer = timm.optim.AdaBelief(parameters, rectify=True, **opt_args)
        case OptimType.ADADELTA:
            optimizer = torch.optim.Adadelta(parameters, **opt_args)
        case OptimType.ADAGRAD:
            opt_args.setdefault("eps", 1e-8)
            optimizer = torch.optim.Adagrad(parameters, **opt_args)
        case OptimType.ADAFACTOR:
            optimizer = timm.optim.Adafactor(parameters, **opt_args)
        case OptimType.ADANP:
            optimizer = timm.optim.Adan(parameters, no_prox=False, **opt_args)
        case OptimType.ADANW:
            optimizer = timm.optim.Adan(parameters, no_prox=True, **opt_args)
        case OptimType.LAMB:
            optimizer = timm.optim.Lamb(parameters, **opt_args)
        case OptimType.LAMBC:
            optimizer = timm.optim.Lamb(parameters, trust_clip=True, **opt_args)
        case OptimType.LARC:
            assert momentum is not None
            optimizer = timm.optim.Lars(parameters, momentum=momentum, trust_clip=True, **opt_args)
        case OptimType.LARS:
            assert momentum is not None
            optimizer = timm.optim.Lars(parameters, momentum=momentum, **opt_args)
        case OptimType.NLARC:
            assert momentum is not None
            optimizer = timm.optim.Lars(parameters, momentum=momentum, trust_clip=True, nesterov=True, **opt_args)
        case OptimType.NLARS:
            assert momentum is not None
            optimizer = timm.optim.Lars(parameters, momentum=momentum, nesterov=True, **opt_args)
        case OptimType.MADGRAD:
            assert momentum is not None
            optimizer = timm.optim.MADGRAD(parameters, momentum=momentum, **opt_args)
        case OptimType.MADGRADW:
            assert momentum is not None
            optimizer = timm.optim.MADGRAD(parameters, momentum=momentum, decoupled_decay=True, **opt_args)
        case OptimType.NOVOGRAD:
            optimizer = timm.optim.NvNovoGrad(parameters, **opt_args)
        case OptimType.RMSPROP:
            assert momentum is not None
            optimizer = torch.optim.RMSprop(parameters, alpha=0.9, momentum=momentum, **opt_args)
        case OptimType.RMSPROPTF:
            assert momentum is not None
            optimizer = timm.optim.RMSpropTF(parameters, alpha=0.9, momentum=momentum, **opt_args)
        case OptimType.LION:
            opt_args.pop("eps", None)
            optimizer = timm.optim.Lion(parameters, **opt_args)
        case OptimType.ADAHESSIAN:
            optimizer = timm.optim.Adahessian(parameters, **opt_args)
        case _:
            raise ValueError(f"Optimizer {opt} not supported in default dispatch.")

    return optimizer


def _create_apex_optimizer(opt: OptimType, parameters: Params, /, momentum: float, **opt_args) -> torch.optim.Optimizer:
    """
    Create an optimizer using the Apex package.

    The Apex package provides a collection of PyTorch optimizers that can be used to speed up training and reduce memory
    usage. These optimizers are designed to work well on modern hardware, such as GPUs and TPUs, and can be used to train
    large-scale deep learning models.

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
    import apex.optimizers

    if not torch.cuda.is_available():
        raise RuntimeError("apex optimizers require CUDA to be available.")

    match opt:
        case OptimType.SGD:
            assert momentum is not None
            opt_args.pop("eps", None)
            optimizer = apex.optimizers.FusedSGD(parameters, momentum=momentum, nesterov=True, **opt_args)
        case OptimType.MOMENTUM:
            assert momentum is not None
            opt_args.pop("eps", None)
            optimizer = apex.optimizers.FusedSGD(parameters, momentum=momentum, nesterov=False, **opt_args)
        case OptimType.ADAM:
            optimizer = apex.optimizers.FusedAdam(parameters, adam_w_mode=False, **opt_args)
        case OptimType.ADAMW:
            optimizer = apex.optimizers.FusedAdam(parameters, adam_w_mode=True, **opt_args)
        case OptimType.LAMB:
            optimizer = apex.optimizers.FusedLAMB(parameters, **opt_args)
        case OptimType.NOVOGRAD:
            opt_args.setdefault("betas", (0.95, 0.98))
            optimizer = apex.optimizers.FusedNovoGrad(parameters, **opt_args)
        case _:
            raise NotImplementedError(f"Optimizer {opt} not supported in 'apex' package.")
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
    import bitsandbytes as bnb

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
            raise NotImplementedError(f"Optimizer {opt} not supported in 'bitsandbytes' package.")
    return optimizer


def _list_optimizer_params(
    model_or_params: ModelOrParams,
    weight_decay: float,
    *,
    filter_bias_and_bn: bool,
    layer_decay: float | None = None,
    param_group_fn: T.Callable | None = None,
) -> tuple[list[nn.Parameter], float]:
    if not isinstance(model_or_params, nn.Module):
        assert isinstance(model_or_params, T.Iterable)
        parameters = list(model_or_params)
        assert all(isinstance(p, nn.Parameter) for p in parameters), "Invalid parameter type in model_or_params"

        logging.debug(f"Optimizer received a list of {len(parameters)} parameters, returning them as-is...")
        return parameters, weight_decay

    # Check whether the model has a `no_weight_decay` method
    no_weight_decay = {}
    if hasattr(model_or_params, "no_weight_decay"):
        no_weight_decay = model_or_params.no_weight_decay()  # type: ignore

    # Extract parameters from model
    if param_group_fn:
        parameters = param_group_fn(model_or_params)
    elif layer_decay is not None:
        parameters = timm.optim.optim_factory.param_groups_layer_decay(
            model_or_params,
            weight_decay=weight_decay,
            layer_decay=layer_decay,
            no_weight_decay_list=no_weight_decay,  # type: ignore
        )
        weight_decay = 0.0
    elif weight_decay and filter_bias_and_bn:
        parameters = timm.optim.optim_factory.param_groups_weight_decay(model_or_params, weight_decay, no_weight_decay)
        weight_decay = 0.0
    else:
        parameters = model_or_params.parameters()

    parameters = list(parameters)

    logging.debug(
        f"Optimizer received a model, extracted {len(parameters)} parameters and {weight_decay} weight decay."
    )

    return parameters, weight_decay
# def get_default_optimizer_params(
#     model: torch.nn.Module,
#     base_lr: Optional[float] = None,
#     weight_decay: Optional[float] = None,
#     weight_decay_norm: Optional[float] = None,
#     bias_lr_factor: Optional[float] = 1.0,
#     weight_decay_bias: Optional[float] = None,
#     lr_factor_func: Optional[Callable] = None,
#     overrides: Optional[Dict[str, Dict[str, float]]] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Get default param list for optimizer, with support for a few types of
#     overrides. If no overrides needed, this is equivalent to `model.parameters()`.

#     Args:
#         base_lr: lr for every group by default. Can be omitted to use the one in optimizer.
#         weight_decay: weight decay for every group by default. Can be omitted to use the one
#             in optimizer.
#         weight_decay_norm: override weight decay for params in normalization layers
#         bias_lr_factor: multiplier of lr for bias parameters.
#         weight_decay_bias: override weight decay for bias parameters.
#         lr_factor_func: function to calculate lr decay rate by mapping the parameter names to
#             corresponding lr decay rate. Note that setting this option requires
#             also setting ``base_lr``.
#         overrides: if not `None`, provides values for optimizer hyperparameters
#             (LR, weight decay) for module parameters with a given name; e.g.
#             ``{"embedding": {"lr": 0.01, "weight_decay": 0.1}}`` will set the LR and
#             weight decay values for all module parameters named `embedding`.

#     For common detection models, ``weight_decay_norm`` is the only option
#     needed to be set. ``bias_lr_factor,weight_decay_bias`` are legacy settings
#     from Detectron1 that are not found useful.

#     Example:
#     ::
#         torch.optim.SGD(get_default_optimizer_params(model, weight_decay_norm=0),
#                        lr=0.01, weight_decay=1e-4, momentum=0.9)
#     """
#     if overrides is None:
#         overrides = {}
#     defaults = {}
#     if base_lr is not None:
#         defaults["lr"] = base_lr
#     if weight_decay is not None:
#         defaults["weight_decay"] = weight_decay
#     bias_overrides = {}
#     if bias_lr_factor is not None and bias_lr_factor != 1.0:
#         # NOTE: unlike Detectron v1, we now by default make bias hyperparameters
#         # exactly the same as regular weights.
#         if base_lr is None:
#             raise ValueError("bias_lr_factor requires base_lr")
#         bias_overrides["lr"] = base_lr * bias_lr_factor
#     if weight_decay_bias is not None:
#         bias_overrides["weight_decay"] = weight_decay_bias
#     if len(bias_overrides):
#         if "bias" in overrides:
#             raise ValueError("Conflicting overrides for 'bias'")
#         overrides["bias"] = bias_overrides
#     if lr_factor_func is not None:
#         if base_lr is None:
#             raise ValueError("lr_factor_func requires base_lr")
#     norm_module_types = (
#         torch.nn.BatchNorm1d,
#         torch.nn.BatchNorm2d,
#         torch.nn.BatchNorm3d,
#         torch.nn.SyncBatchNorm,
#         # NaiveSyncBatchNorm inherits from BatchNorm2d
#         torch.nn.GroupNorm,
#         torch.nn.InstanceNorm1d,
#         torch.nn.InstanceNorm2d,
#         torch.nn.InstanceNorm3d,
#         torch.nn.LayerNorm,
#         torch.nn.LocalResponseNorm,
#     )
#     params: List[Dict[str, Any]] = []
#     memo: Set[torch.nn.parameter.Parameter] = set()
#     for module_name, module in model.named_modules():
#         for module_param_name, value in module.named_parameters(recurse=False):
#             if not value.requires_grad:
#                 continue
#             # Avoid duplicating parameters
#             if value in memo:
#                 continue
#             memo.add(value)

#             hyperparams = copy.copy(defaults)
#             if isinstance(module, norm_module_types) and weight_decay_norm is not None:
#                 hyperparams["weight_decay"] = weight_decay_norm
#             if lr_factor_func is not None:
#                 hyperparams["lr"] *= lr_factor_func(f"{module_name}.{module_param_name}")

#             hyperparams.update(overrides.get(module_param_name, {}))
#             params.append({"params": [value], **hyperparams})
#     return reduce_param_groups(params)


# def _expand_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     # Transform parameter groups into per-parameter structure.
#     # Later items in `params` can overwrite parameters set in previous items.
#     ret = defaultdict(dict)
#     for item in params:
#         assert "params" in item
#         cur_params = {x: y for x, y in item.items() if x != "params" and x != "param_names"}
#         if "param_names" in item:
#             for param_name, param in zip(item["param_names"], item["params"]):
#                 ret[param].update({"param_names": [param_name], "params": [param], **cur_params})
#         else:
#             for param in item["params"]:
#                 ret[param].update({"params": [param], **cur_params})
#     return list(ret.values())


# def reduce_param_groups(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     # Reorganize the parameter groups and merge duplicated groups.
#     # The number of parameter groups needs to be as small as possible in order
#     # to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
#     # of using a parameter_group per single parameter, we reorganize the
#     # parameter groups and merge duplicated groups. This approach speeds
#     # up multi-tensor optimizer significantly.
#     params = _expand_param_groups(params)
#     groups = defaultdict(list)  # re-group all parameter groups by their hyperparams
#     for item in params:
#         cur_params = tuple((x, y) for x, y in item.items() if x != "params" and x != "param_names")
#         groups[cur_params].append({"params": item["params"]})
#         if "param_names" in item:
#             groups[cur_params][-1]["param_names"] = item["param_names"]

#     ret = []
#     for param_keys, param_values in groups.items():
#         cur = {kv[0]: kv[1] for kv in param_keys}
#         cur["params"] = list(
#             itertools.chain.from_iterable([params["params"] for params in param_values])
#         )
#         if len(param_values) > 0 and "param_names" in param_values[0]:
#             cur["param_names"] = list(
#                 itertools.chain.from_iterable([params["param_names"] for params in param_values])
#             )
#         ret.append(cur)
#     return ret