import typing as T

import torch
import torch.compiler
import torch.fx
import typing_extensions as TX
from torch import nn

from unipercept.types import Tensor
from unipercept.utils.inspect import locate_object

type ActivationFactory = T.Callable[[], nn.Module]
type ActivationSpec = ActivationFactory | type[nn.Module] | nn.Module | None | str


def get_activation(spec: ActivationSpec) -> nn.Module:
    """
    Resolve an activation module from a string, a factory function or a module instance.

    Parameters
    ----------
    activation
        A string, a factory function or an instance of an activation module.
    """
    # Check if no activation is desired
    if spec is None:
        return nn.Identity()

    # Check whether activation is provided as a string or module path
    if isinstance(spec, str):
        spec, *spec_args = spec.split(":", 1) if ":" in spec else (spec, None)
        if len(spec_args) > 0:
            args = map(str.strip, T.cast(list[str], spec_args))
        else:
            args = iter([])
        match spec.lower().strip():
            case "relu":
                return nn.ReLU(
                    inplace=next(args, "").lower() == "inplace",
                )
            case "relu6":
                return nn.ReLU6(
                    inplace=next(args, "").lower() == "inplace",
                )
            case "leaky-relu":
                return nn.LeakyReLU(
                    negative_slope=float(next(args, "0.01")),
                    inplace=next(args, "").lower() == "inplace",
                )
            case "gelu":
                return nn.GELU(
                    approximate=next(args, "none").lower(),
                )
            case "prelu":
                return PReLU(
                    num_parameters=int(next(args, "1")),
                    init=float(next(args, "0.25")),
                )
            case "silu":
                return nn.SiLU()
            case "swish":
                return nn.SiLU()
            case "mish":
                return nn.Mish()
            case "sigmoid":
                return nn.Sigmoid()
            case "tanh":
                return nn.Tanh()
            case "softmax":
                return nn.Softmax(
                    dim=int(next(args, "-1")),
                )
            case "softplus":
                return nn.Softplus()
            case "softsign":
                return nn.Softsign()
            case "threshold":
                if len(spec_args) < 2:
                    msg = (
                        r"Threshold activation requires at least two arguments. "
                        r"Provide them as `theshold:a:b` where `a` is the thehsold and `b` is the value."
                    )
                    raise ValueError(msg)
                return nn.Threshold(
                    threshold=float(next(args)),
                    value=float(next(args)),
                    inplace=next(args, "").lower() == "inplace",
                )
            case "identity":
                return nn.Identity()
            case "none":
                return nn.Identity()
            case _:
                spec = locate_object(spec)

    # If already a module instance, return that instance directly
    if isinstance(spec, nn.Module):
        return spec
    if callable(spec):
        return spec()
    raise ValueError(f"Cannot resolve value as an activation module: {spec}")


class ApproxTanhGELU(nn.Module):
    r"""
    An approximate verison of the GELU defined as
    :math:`GELU(x) = \frac{x}{2}(tanh(\sqrt{2/\pi}(x+0.044715x^3)))`.
    """

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(x, approximate="tanh")


class ApproxSigmoidGELU(nn.Module):
    r"""
    An approximate verison of the GELU defined as
    :math:`GELU(x) = x\sigma(1.702x)`.
    """

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        return x * (x * 1.702).sigmoid()


class InplaceReLU(nn.Module):
    """
    A ReLU activation function that performs the operation in-place.
    """

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        inplace = not (
            torch.jit.is_scripting()
            or torch.compiler.is_compiling()
            or torch.compiler.is_dynamo_compiling()
        )
        return nn.functional.relu(x, inplace=inplace)


class InplaceReLU6(nn.Module):
    """
    A ReLU6 activation function that performs the operation in-place.
    """

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        inplace = not (
            torch.jit.is_scripting()
            or torch.compiler.is_compiling()
            or torch.compiler.is_dynamo_compiling()
        )
        return nn.functional.hardtanh(x, 0.0, 6.0, inplace=inplace)


ReLU = nn.ReLU
ReLU6 = nn.ReLU6
LeakyReLU = nn.LeakyReLU
GELU = nn.GELU
SiLU = nn.SiLU
Mish = nn.Mish
Sigmoid = nn.Sigmoid
Tanh = nn.Tanh
Softmax = nn.Softmax
Softsign = nn.Softsign
Threshold = nn.Threshold
Identity = nn.Identity


class PReLU(nn.Module):
    """
    Parametric ReLU activation function. This wraps the :func:`torch.nn.PReLU` module,
    but renames the ``weight`` parameter to ``param`` such that the optimizer can
    differentiate between the network weights that are learned with decay and
    the second-class parameters that are learned without decay.
    """

    __constants__ = ["num_parameters"]
    num_parameters: int

    def __init__(
        self, num_parameters: int = 1, init: float = 0.25, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.param = nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.param, self.init)

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.prelu(input, self.param)

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}"


############
# Softplus #
############

_SOFTPLUS_DEFAULT_BETA: T.Final[float] = 1.0
_SOFTPLUS_DEFAULT_THRESHOLD: T.Final[float] = 20.0
_SOFTPLUS_DEFAULT_EPS: T.Final[float] = torch.finfo(torch.float32).resolution


def softplus(
    x: Tensor,
    beta: float = _SOFTPLUS_DEFAULT_BETA,
    threshold: float = _SOFTPLUS_DEFAULT_THRESHOLD,
) -> Tensor:
    """
    Softplus function with defaults that match the inverse defined in this module.

    Parameters
    ----------
    x
        Input tensor.
    beta
        Coefficient for the softplus function.
    threshold
        Threshold for the softplus function. The result is linear for (x * beta) > threshold.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return nn.functional.softplus(x, beta=beta, threshold=threshold)


class Softplus(nn.Module):
    """
    Numerically stable softplus function. See :func:`softplus`.
    """

    def __init__(
        self,
        beta: float = _SOFTPLUS_DEFAULT_BETA,
        threshold: float = _SOFTPLUS_DEFAULT_THRESHOLD,
    ):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        return softplus(x, beta=self.beta, threshold=self.threshold)


def inverse_softplus(
    y: Tensor,
    beta: float = _SOFTPLUS_DEFAULT_BETA,
    threshold: float = _SOFTPLUS_DEFAULT_THRESHOLD,
    eps=1e-6,
) -> Tensor:
    """
    Numerically stable inverse of the softplus function.

    Note that this can only approximate the inverse of values that
    are greater than zero, as the softplus function is not invertible
    for negative values.

    Parameters
    ----------
    x
        Input tensor.
    beta
        Coefficient for the softplus function.

    Returns
    -------
    Tensor
        Output tensor.
    """
    return y.where(y >= threshold, y + (-(-y * beta).expm1()).log() / beta)


class InverseSoftplus(nn.Module):
    """
    Numerically stable inverse of the softplus function. See :func:`inverse_softplus`.
    """

    def __init__(
        self,
        beta: float = _SOFTPLUS_DEFAULT_BETA,
        threshold: float = _SOFTPLUS_DEFAULT_THRESHOLD,
        eps: float = _SOFTPLUS_DEFAULT_EPS,
    ):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.eps = eps

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        return inverse_softplus(
            x, beta=self.beta, threshold=self.threshold, eps=self.eps
        )
