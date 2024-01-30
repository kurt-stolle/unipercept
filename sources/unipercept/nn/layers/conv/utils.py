"""Various utility functions and classes for working with convolutional layers."""

from __future__ import annotations

import enum
import functools
import inspect
import logging
import math
import typing as T

import torch
import torch.nn as nn
from typing_extensions import override

from unipercept.nn.layers.activation import ActivationSpec, get_activation
from unipercept.nn.layers.norm import NormSpec, get_norm
from unipercept.utils.abbreviate import short_name
from unipercept.utils.function import to_2tuple

__all__ = [
    "with_norm_activation",
    "NormActivationMixin",
    "with_padding_support",
    "Padding",
    "PaddingMixin",
]


_OUTPUT_CHANNEL_PROPERTIES = ["out_channels", "out_dims"]


@T.overload
def get_output_channels(
    mod: nn.Module, *, use_weight: bool = True, none_ok: bool = False
) -> int:
    ...


@T.overload
def get_output_channels(
    mod: nn.Module, *, use_weight: bool = True, none_ok: bool = True
) -> int | None:
    ...


def get_output_channels(
    mod: nn.Module, *, use_weight=True, none_ok=False
) -> int | None:
    """Returns the number of output channels of a module."""
    for prop in _OUTPUT_CHANNEL_PROPERTIES:
        if not hasattr(mod, prop):
            continue
        return getattr(mod, prop)

    if use_weight and hasattr(mod, "weight"):
        return mod.weight.shape[0]

    if none_ok is True:
        return None
    else:
        raise ValueError(f"Module {mod} does not have an output channel property!")


# ---------------------- #
# Norm and activation    #
# ---------------------- #


@torch.jit.ignore()
def _init_sequential_norm_activation(
    cls,
    *args,
    norm: T.Optional[NormSpec],
    activation: T.Optional[ActivationSpec],
    **kwargs,
) -> nn.Sequential:
    """Adds optional `norm` and `activation` parameters."""

    act = get_activation(activation)

    seq = _init_sequential_norm(cls, *args, norm=norm, **kwargs)
    seq.add_module(short_name(act), act)

    return seq


@torch.jit.ignore()
def _init_sequential_norm(
    cls, *args, norm: T.Optional[NormSpec], bias: T.Optional[bool] = None, **kwargs
) -> nn.Sequential:
    """
    Adds an optional ``norm`` parameter, which defaults to LayerNormCHW.
    Setting the ``bias`` parameter to ``None`` will determine whether a bias should
    be added based on the ``norm`` parameter. If ``norm`` is ``None``, ``bias`` will
    be set to ``True``.
    """

    bias = bias if bias is not None else norm is None
    assert isinstance(bias, bool)

    mod = cls(*args, bias=bias, **kwargs)
    num_channels = get_output_channels(mod, none_ok=False)
    nrm = get_norm(norm, num_channels)

    seq = nn.Sequential()
    seq.add_module(short_name(mod), mod)
    seq.add_module(short_name(nrm), nrm)
    return seq


@torch.jit.ignore()
def _init_sequential_activation(
    cls, *args, activation: T.Optional[ActivationSpec] = None, **kwargs
) -> nn.Sequential:
    """Adds an optional `activation` parameter."""
    mod = cls(*args, **kwargs)
    act = get_activation(activation)

    seq = nn.Sequential()
    seq.add_module(short_name(mod), mod)
    seq.add_module(short_name(act), act)

    return seq


_M = T.TypeVar("_M", bound=nn.Module)


def with_norm_activation(cls: type[_M]) -> type[_M]:
    """Augments a class with norm and activation helpers."""

    cls.with_activation = classmethod(_init_sequential_activation)  # type: ignore
    cls.with_norm = classmethod(_init_sequential_norm)  # type: ignore
    cls.with_norm_activation = classmethod(_init_sequential_norm_activation)  # type: ignore

    return cls


class NormActivationMixin:
    """Mixin for initialization helpers that add norm and activation layers."""

    with_activation = classmethod(_init_sequential_activation)
    with_norm = classmethod(_init_sequential_norm)
    with_norm_activation = classmethod(_init_sequential_norm_activation)


# --------------- #
# Padding support #
# --------------- #


class Padding(enum.StrEnum):
    VALID = enum.auto()
    SAME = enum.auto()


_PaddingInput = T.TypeVar("_PaddingInput", torch.Tensor, int)
_PaddingValue: T.TypeAlias = int | T.Tuple[int, int]
_PaddingParam: T.TypeAlias = str | Padding | _PaddingValue

_InitParams = T.ParamSpec("_InitParams")


_T = T.TypeVar("_T", bound=nn.Module)


def _wrap_init(
    init: T.Callable[T.Concatenate[_T, int, int, int, _InitParams], None]
) -> T.Callable[T.Concatenate[_T, int, int, int, _InitParams], None]:
    inspect.signature(init)

    # Validate signature has first four expected arguments
    # sig_expected = ("self", "in_channels", "out_channels", "kernel_size")
    # for i, (par_expected, par) in enumerate(zip(sig_expected, sig.parameters.values())):
    #     if par_expected != par.name:
    #         raise ValueError(f"Expected parameter {par_expected} at position {i}, got {par.name} instead!")
    #     if par.kind == inspect.Parameter.KEYWORD_ONLY or par.kind == inspect.Parameter.VAR_KEYWORD:
    #         raise ValueError(f"Parameter {par.name} must be a positional argument!")

    # Update signature with new padding parameters
    # par_pad = sig.parameters.get("padding")
    # assert par_pad is not None, "Convolution module must have a padding parameter!"
    # par_pad.replace(annotation=_PaddingParam, default=Padding.SAME)

    @functools.wraps(init)
    def wrapper(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *args: _InitParams.args,
        **kwargs: _InitParams.kwargs,
    ):
        padding: _PaddingParam = kwargs.pop("padding", Padding.SAME)  # type: ignore
        padding, dynamic = _parse_padding(padding, kernel_size, **kwargs)
        assert isinstance(
            padding, (int, tuple)
        ), f"Padding must be an int or a tuple, got {padding}!"

        self.padding_dynamic = dynamic
        init(self, in_channels, out_channels, kernel_size, *args, padding=padding, **kwargs)  # type: ignore

    # wrapper.__signature__ = sig  # type: ignore
    return wrapper  # type: ignore


def _wrap_forward(
    forward: T.Callable[[_T, torch.Tensor], torch.Tensor]
) -> T.Callable[[_T, torch.Tensor], torch.Tensor]:
    @functools.wraps(forward)
    def wrapper(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding_dynamic:
            x = _pad_same(x, self.kernel_size, self.stride, self.dilation)
        x = forward(self, x)
        return x

    return wrapper


def _pad_same_amount(x: int, kernel_size: int, stride: int, dilation: int) -> int:
    """Returns the equivalent to `padding="SAME"` in TensorFlow, which supports strides greater than one."""
    # return torch.clamp(((x / stride).ceil() - 1) * stride + (kernel_size - 1) * dilation + 1 - x, min=0)
    return max(
        (math.ceil(x / stride) - 1) * stride + (kernel_size - 1) * dilation + 1 - x, 0
    )


def _pad_same(
    input: torch.Tensor,
    kernel_size: T.Tuple[int, ...],
    stride: T.Tuple[int, ...],
    dilation: T.Tuple[int, ...] = (1, 1),
    value: float = 0.0,
) -> torch.Tensor:
    ih, iw = input.size()[-2:]
    k0, k1 = kernel_size
    s0, s1 = stride
    d0, d1 = dilation

    pad_h = _pad_same_amount(ih, k0, s0, d0)
    pad_w = _pad_same_amount(iw, k1, s1, d1)

    return nn.functional.pad(
        input,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        value=value,
    )


def _parse_padding(
    padding: _PaddingParam, kernel_size, **kwargs
) -> tuple[_PaddingValue, bool]:
    """ "
    Parse the padding parameter into a value, returns -1 if the padding is not
    able to be statically determined. The second return value is a boolean that
    indicates if the padding is dynamic.
    """

    def __is_padding_static(
        kernel_size: int, stride: int = 1, dilation: int = 1, **_
    ) -> bool:
        return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0

    def __get_padding_static(
        kernel_size: int, stride: int = 1, dilation: int = 1, **_
    ) -> int:
        padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
        return padding

    if isinstance(padding, (int, tuple)):
        return padding, False

    if isinstance(padding, str):
        padding = Padding(padding.lower())

    if isinstance(padding, Padding):
        if padding == Padding.SAME:
            if __is_padding_static(kernel_size, **kwargs):
                return __get_padding_static(kernel_size, **kwargs), False
            return 0, True
        elif padding == Padding.VALID:
            return 0, False
        else:
            raise NotImplementedError(f"Padding type {padding} not implemented yet!")

    if isinstance(padding, T.Iterator) or inspect.isgenerator(padding):
        return tuple(padding), False

    raise ValueError("Unknown padding type %s", type(padding))


def with_padding_support(cls: type[_T]) -> type[_T]:
    cls.__init__ = _wrap_init(cls.__init__)  # type: ignore
    cls.forward = _wrap_forward(cls.forward)  # type: ignore

    return cls  # type: ignore


class PaddingMixin:
    __constants__ = "padding_dynamic"

    @override
    def __init_subclass__(cls, **kwargs) -> None:
        logging.debug(f"Initializing {cls} with padding support...")
        cls.__init__ = _wrap_init(cls.__init__)  # type: ignore
        super().__init_subclass__(**kwargs)

    @torch.jit.unused
    def _parse_padding(
        self, padding: _PaddingParam, kernel_size, **kwargs
    ) -> _PaddingValue:
        """Use during initialization to parse a padding parameter into a value."""
        parsed, self.padding_dynamic = _parse_padding(padding, kernel_size, **kwargs)

        return parsed

    def _padding_forward(
        self,
        x: torch.Tensor,
        ks: T.Tuple[int, ...],
        ss: T.Tuple[int, ...],
        dl: T.Tuple[int, ...],
    ):
        if self.padding_dynamic:
            x = _pad_same(x, ks, ss, dl)
        return x


# -------------------- #
# Other helper methods #
# -------------------- #

_Maybe2Int: T.TypeAlias = int | T.Tuple[int, int]
# -------------------------------- #
# Pooling for convolutional layers #
# -------------------------------- #


def avg_pool2d_same(
    x,
    kernel_size: T.Tuple[int, int],
    stride: T.Tuple[int, int],
    ceil_mode: bool = False,
    count_include_pad: bool = True,
):
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = _pad_same(x, kernel_size, stride)
    return nn.functional.avg_pool2d(
        x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad
    )


class AvgPool2dSame(nn.AvgPool2d):
    """Tensorflow like 'SAME' wrapper for 2D average pooling."""

    def __init__(
        self,
        kernel_size: _Maybe2Int,
        stride: _Maybe2Int | None = None,
        ceil_mode=False,
        count_include_pad=True,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride) if stride is not None else None
        super().__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    @override
    def forward(self, x):
        return avg_pool2d_same(
            x, self.kernel_size, self.stride, self.ceil_mode, self.count_include_pad
        )


def max_pool2d_same(
    x,
    kernel_size: T.Tuple[int, int],
    stride: T.Tuple[int, int],
    dilation: T.Tuple[int, int] = (1, 1),
    ceil_mode: bool = False,
) -> torch.Tensor:
    x = _pad_same(x, kernel_size, stride, value=-float("inf"))
    x = nn.functional.max_pool2d(x, kernel_size, stride, (0, 0), dilation, ceil_mode)
    return x


class MaxPool2dSame(nn.MaxPool2d):
    """Tensorflow like 'SAME' wrapper for 2D max pooling."""

    def __init__(
        self,
        kernel_size: _Maybe2Int,
        stride: _Maybe2Int | None = None,
        dilation=1,
        ceil_mode=False,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride) if stride is not None else None
        dilation = to_2tuple(dilation)
        super().__init__(kernel_size, stride, (0, 0), dilation, ceil_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return max_pool2d_same(
            x, self.kernel_size, self.stride, self.dilation, self.ceil_mode
        )


class MaxPool2d(nn.Module):
    def __new__(
        cls, kernel_size: _Maybe2Int, stride=None, **kwargs
    ) -> nn.MaxPool2d | MaxPool2dSame:
        stride = stride or kernel_size
        padding = kwargs.pop("padding", "")
        padding, is_dynamic = _parse_padding(
            padding, kernel_size, stride=stride, **kwargs
        )

        if is_dynamic:
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)


class AvgPool2d(nn.Module):
    def __new__(
        cls, kernel_size: _Maybe2Int, stride=None, **kwargs
    ) -> nn.AvgPool2d | AvgPool2dSame:
        stride = stride or kernel_size
        padding = kwargs.pop("padding", "")
        padding, is_dynamic = _parse_padding(
            padding, kernel_size, stride=stride, **kwargs
        )

        if is_dynamic:
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)


POOLING_LAYERS = {
    "max": MaxPool2d,
    "avg": AvgPool2d,
}
