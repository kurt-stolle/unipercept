from __future__ import annotations

import typing as T

import torch.nn as nn

from unipercept.utils.inspect import locate_object

ActivationFactory: T.TypeAlias = T.Callable[[], nn.Module]
ActivationSpec: T.TypeAlias = ActivationFactory | nn.Module | None | str


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
            case "leaky-relu":
                return nn.LeakyReLU(
                    negative_slope=float(next(args, "0.01")),
                    inplace=next(args, "").lower() == "inplace",
                )
            case "gelu":
                return nn.GELU(
                    approximate=next(args, "none").lower(),
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
    elif callable(spec):
        return spec()
    else:
        raise ValueError(f"Cannot resolve value as an activation module: {spec}")


class InplaceReLU(nn.ReLU):
    """
    A ReLU activation function that performs the operation in-place.
    """

    def __init__(self):
        super().__init__(inplace=True)
