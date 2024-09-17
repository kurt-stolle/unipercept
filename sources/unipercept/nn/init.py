import copy
import enum as E
import math
import typing as T
import warnings
from collections.abc import Callable, Sequence
from types import EllipsisType

import regex as re
import torch
import torch.overrides
from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from torch import nn
from torch.nn.init import (
    _calculate_correct_fan,
    _calculate_fan_in_and_fan_out,
    _no_grad_trunc_normal_,
)
from torch.nn.modules.conv import _ConvNd

from unipercept.types import Tensor

_NONLINEARITY_TO_CANONICAL_MAP: T.OrderedDict[
    type[nn.Module] | re.Pattern, tuple[str, T.Any]
] = T.OrderedDict(
    {
        re.compile(".*leaky_?relu.*"): ("leaky_relu", lambda m: m.negative_slope),
        re.compile(".*selu.*"): ("selu", None),
        re.compile(".*tanh.*"): ("relu", None),
        re.compile(".*sigmoid.*"): ("sigmoid", None),
        re.compile(".*conv.*"): ("linear", None),
        re.compile(".*linear.*"): ("linear", None),
        re.compile(".*elu.*"): ("relu", None),
        re.compile(".*identity.*"): ("linear", None),
    }
)


def _infer_namegainparam(module: nn.Module) -> tuple[str, float, T.Any]:
    """
    Get the name of the nonlinearity and its gain.
    """

    if isinstance(module, type):
        msg = "Cannot determine nonlinearity name and gain for module class. Pass an instance instead."
        raise ValueError(msg)

    for key, (name, param) in _NONLINEARITY_TO_CANONICAL_MAP.items():
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

    if callable(param):
        param = param(module)
    gain = nn.init.calculate_gain(name, param=param)

    return name, gain, param


def init_wrap(
    modules: Sequence[type[nn.Module]],
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

############################
# Truncated Kaiming/Xavier #
############################

zeros_ = nn.init.zeros_
ones_ = nn.init.ones_
dirac_ = nn.init.dirac_
sparse_ = nn.init.sparse_
eye_ = nn.init.eye_
kaiming_normal_ = nn.init.kaiming_normal_
kaiming_uniform_ = nn.init.kaiming_uniform_
xavier_normal_ = nn.init.xavier_normal_
xavier_uniform_ = nn.init.xavier_uniform_
orthogonal_ = nn.init.orthogonal_


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, bound: float = 2.0
) -> Tensor:
    a = mean - bound * std
    b = mean + bound * std

    assert a < mean < b, f"Invalid truncation bounds: {a} < {mean} < {b}"
    return _no_grad_trunc_normal_(tensor, mean, std, a=a, b=b)


def trunc_xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_trunc_normal_(
        tensor, 0.0, std, a=-2 * std, b=2 * std, generator=generator
    )


def trunc_kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu",
    generator: torch.Generator | None = None,
):
    if 0 in tensor.shape:
        warnings.warn("Initializing zero-element tensors is a no-op")
        return tensor
    fan = _calculate_correct_fan(tensor, mode)
    gain = nn.init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)

    return _no_grad_trunc_normal_(
        tensor, 0.0, std, a=-2 * std, b=2 * std, generator=generator
    )


#######################
# Init by enumeration #
#######################


def discover_weights(module: nn.Module, *, strict=False) -> list[str]:
    """
    Discover the weight names of a module based on the class type/name.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return ["weight"]
    if not strict and hasattr("module", "weight"):
        return ["weight"]
    return []


def discover_biases(module: nn.Module, *, strict=False) -> list[str]:
    """
    Discover the bias names of a module based on the class type/name.
    """
    if isinstance(
        module,
        (
            _ConvNd,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose1d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        ),
    ):
        return ["bias"]
    if not strict and hasattr("module", "bias"):
        return ["bias"]
    return []


class InitMode(E.StrEnum):
    DEFAULT = E.auto()
    NONE = E.auto()
    ZEROS = E.auto()
    ONES = E.auto()
    DIRAC = E.auto()
    SPARSE = E.auto()
    EYE = E.auto()
    KAIMING_NORMAL = E.auto()
    KAIMING_UNIFORM = E.auto()
    XAVIER_NORMAL = E.auto()
    XAVIER_UNIFORM = E.auto()
    TRUNC_NORMAL = E.auto()
    ORTHOGONAL = E.auto()
    C2_MSRA = E.auto()
    C2_XAVIER = E.auto()
    C2_NORMAL = E.auto()


type InitSpec = dict[str, T.Any]


def init_module_(
    module: nn.Module,
    mode: InitMode | str,
    /,
    *,
    weights: T.Sequence[str | nn.Parameter] | None = None,
    biases: T.Sequence[str | nn.Parameter] | None = None,
    **kwargs,
) -> None:
    if weights is None:
        weights = discover_weights(module)
    weights_list = []
    for p in weights:
        if isinstance(p, str):
            p = getattr(module, p, None)
        if p is None:
            continue
        assert isinstance(p, (nn.Parameter, torch.Tensor))
        weights_list.append(p)

    if biases is None:
        biases = discover_biases(module)
    biases_list = []
    for p in biases:
        if isinstance(p, str):
            p = getattr(module, p, None)
        if p is None:
            continue
        assert isinstance(p, (nn.Parameter, torch.Tensor))
        biases_list.append(p)

    if len(biases_list) == 0 and len(weights_list) == 0:
        return

    for attr in ("activation", "act", "nonlinearity"):
        if hasattr(module, attr):
            nonlin = getattr(module, attr)
            break
    else:
        nonlin = module

    nonlin_name, gain, param = _infer_namegainparam(nonlin)

    match mode:
        case InitMode.KAIMING_NORMAL | InitMode.KAIMING_UNIFORM:
            kwargs.setdefault("nonlinearity", nonlin_name)
            if param is not None:
                kwargs.setdefault("a", param)
        case InitMode.XAVIER_NORMAL | InitMode.XAVIER_UNIFORM | InitMode.ORTHOGONAL:
            kwargs.setdefault("gain", gain)
        case _:
            pass

    init_parameters_(mode, weights_list, biases_list, **kwargs)


def init_recursive_(
    module: nn.Module,
    mode: InitMode | str,
    /,
    *,
    whitelist: tuple[type[nn.Module], ...] | None = None,
    blacklist: tuple[type[nn.Module], ...] | None = None,
    **kwargs,
):
    def apply(module: nn.Module):
        if whitelist is not None and not isinstance(module, whitelist):
            return
        if blacklist is not None and isinstance(module, blacklist):
            return
        init_module_(
            module,
            mode,
            **kwargs,
        )

    module.apply(apply)


def init_parameters_(
    mode: InitMode | str,
    weights: T.Iterable[nn.Parameter],
    biases: T.Iterable[nn.Parameter],
    /,
    *,
    generator: torch.Generator | None = None,
    **kwargs,
):
    match mode:
        case InitMode.NONE:
            return
        case InitMode.DEFAULT:
            pass
        case InitMode.ZEROS:
            for weight in weights:
                nn.init.zeros_(weight)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.ONES:
            for weight in weights:
                nn.init.ones_(weight)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.DIRAC:
            groups = kwargs.pop("groups", 1)
            for weight in weights:
                nn.init.dirac_(weight, groups=groups)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.SPARSE:
            sparsity = kwargs.pop("sparsity", 0.1)
            std = kwargs.pop("std", 0.01)
            for weight in weights:
                nn.init.sparse_(weight, sparsity=sparsity, std=std)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.EYE:
            for weight in weights:
                nn.init.eye_(weight)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.KAIMING_NORMAL:
            nonlinearity = kwargs.pop("nonlinearity", "relu")
            if nonlinearity not in ("relu", "leaky_relu") and not kwargs.pop(
                "force", False
            ):
                msg = f"Unsupported nonlinearity for Kaiming initialization: {nonlinearity}, pass `force=True` to ignore"
                raise ValueError(msg)
            mode = kwargs.pop("mode", "fan_in")
            a = kwargs.pop("a", 0)

            if kwargs.pop("truncate", True):
                fn = trunc_kaiming_normal_
            else:
                fn = nn.init.kaiming_normal_
            for weight in weights:
                fn(
                    weight,
                    generator=generator,
                    nonlinearity=nonlinearity,
                    mode=mode,
                    a=a,
                )
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.KAIMING_UNIFORM:
            nonlinearity = kwargs.pop("nonlinearity", "relu")
            if nonlinearity not in ("relu", "leaky_relu") and not kwargs.pop(
                "force", False
            ):
                msg = f"Unsupported nonlinearity for Kaiming initialization: {nonlinearity}"
                raise ValueError(msg)
            mode = kwargs.pop("mode", "fan_in")
            a = kwargs.pop("a", 0)

            for weight in weights:
                nn.init.kaiming_uniform_(
                    weight,
                    generator=generator,
                    nonlinearity=nonlinearity,
                    mode=mode,
                    a=a,
                )
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.XAVIER_NORMAL:
            gain = kwargs.pop("gain", 1.0)
            if kwargs.pop("truncate", True):
                fn = trunc_xavier_normal_
            else:
                fn = nn.init.xavier_normal_
            for weight in weights:
                nn.init.xavier_normal_(weight, generator=generator, gain=gain)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.XAVIER_UNIFORM:
            gain = kwargs.pop("gain", 1.0)
            for weight in weights:
                nn.init.xavier_uniform_(weight, generator=generator, gain=gain)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.ORTHOGONAL:
            gain = kwargs.pop("gain", 1.0)
            for weight in weights:
                nn.init.orthogonal_(weight, generator=generator, gain=gain)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.TRUNC_NORMAL:
            mean: float = kwargs.pop("mean", 0.0)
            std: float = kwargs.pop("std", 1.0)
            a: float = kwargs.pop("a", mean - 2.0 * std)
            b: float = kwargs.pop("b", mean + 2.0 * std)
            for weight in weights:
                nn.init.trunc_normal_(
                    weight, generator=generator, mean=mean, std=std, a=a, b=b
                )
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.C2_MSRA:
            for weight in weights:
                nn.init.kaiming_normal_(
                    weight, generator=generator, mode="fan_out", nonlinearity="relu"
                )
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.C2_XAVIER:
            for weight in weights:
                nn.init.kaiming_uniform_(weight, generator=generator, a=1)
            for bias in biases:
                nn.init.zeros_(bias)
        case InitMode.C2_NORMAL:
            for weight in weights:
                nn.init.kaiming_normal_(weight, generator=generator, a=1)
            for bias in biases:
                nn.init.zeros_(bias)
        case _:
            msg = f"Unknown initialization mode: {mode}"
            raise NotImplementedError(msg)

    if len(kwargs) > 0:
        msg = f"Unused initialization arguments: {kwargs}"
        raise ValueError(msg)


#####################
# Init module mixin #
#####################


class InitMixin:
    r"""
    Adds an init keyword argument to the constructor and a reset_parameters method to
    apply the initialization.

    Useful for configuration files.
    """

    init_mode: T.Final[str]
    init_spec: T.Final[dict[str, T.Any]]

    def __init__(
        self,
        *args,
        init_mode: InitMode | str | None = None,
        init_spec: dict[str, T.Any] | None = None,
        **kwargs,
    ):
        self.init_mode = (
            InitMode(init_mode).value
            if init_mode is not None
            else InitMode.DEFAULT.value
        )
        self.init_spec = (
            {str(k): copy.deepcopy(v) for k, v in init_spec.items()}
            if init_spec is not None
            else {}
        )
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if self.init_mode == InitMode.DEFAULT:
            if hasattr(super(), "reset_parameters"):
                super().reset_parameters()
            elif hasattr(self, "init_default"):
                self.init_default()
            elif hasattr(self, "_reset_parameters"):
                self._reset_parameters()
        else:
            init_recursive_(self, self.init_mode, **self.init_spec)
