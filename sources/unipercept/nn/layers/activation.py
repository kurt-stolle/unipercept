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

    # Check whether activation is provided as a path
    if isinstance(spec, str):
        spec = locate_object(spec)

    # If already a module instance, return that instance directly
    if isinstance(spec, nn.Module):
        return spec
    elif callable(spec):
        return spec()
    else:
        raise ValueError(f"Cannot resolve value as an activation module: {spec}")
