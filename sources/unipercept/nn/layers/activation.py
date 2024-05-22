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
        match spec:
            case "relu":
                spec = nn.ReLU
            case "relu-inplace":
                spec = InplaceReLU
            case "leaky-relu":
                spec = nn.LeakyReLU
            case "gelu":
                spec = nn.GELU
            case "silu":
                spec = nn.SiLU
            case "swish":
                spec = nn.SiLU
            case "mish":
                spec = nn.Mish
            case "sigmoid":
                spec = nn.Sigmoid
            case "tanh":
                spec = nn.Tanh
            case "softplus":
                spec = nn.Softplus
            case "softsign":
                spec = nn.Softsign
            case "identity":
                spec = nn.Identity
            case "none":
                spec = nn.Identity
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
