"""
General helper functions for evaluators.
"""

from __future__ import annotations

import typing as T

import torch
import torch.special

__all__ = ["DTYPE_INT", "DTYPE_FLOAT", "EPS", "stable_divide", "isin"]

DTYPE_INT: T.Final = torch.int32
DTYPE_FLOAT: T.Final = torch.float32
EPS: T.Final[float] = torch.finfo(DTYPE_FLOAT).eps
POS_INF: T.Final[float] = torch.finfo(DTYPE_FLOAT).max * EPS
NEG_INF: T.Final[float] = torch.finfo(DTYPE_FLOAT).min * EPS


def stable_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Divide two floating point numbers, but avoid division by zero by adding a small epsilon value.

    Examples
    --------
    >>> a = torch.tensor([1, 2, 3], dtype=torch.float64)
    >>> b = torch.tensor([0, 1, 2], dtype=torch.float64)
    >>> stable_divide(a, b)
    tensor([1.0000, 2.0000, 1.5000], dtype=torch.float64)
    """

    # return a.to(dtype=DTYPE_FLOAT) / (b.to(dtype=DTYPE_FLOAT) + EPS)  # .clamp_min_(EPS)

    a = a.to(dtype=DTYPE_FLOAT)
    b = b.to(dtype=DTYPE_FLOAT)
    return torch.nan_to_num(a / b, nan=0.0, posinf=POS_INF, neginf=NEG_INF)


def nonzero_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Divide two floating point numbers, but set the result to zero if the denominator is zero.
    """

    a = a.to(dtype=DTYPE_FLOAT)
    b = b.to(dtype=DTYPE_FLOAT)
    return torch.where(b == 0, torch.zeros_like(a), stable_divide(a, b))


def isin(arr: torch.Tensor, values: list) -> torch.Tensor:
    """Check if all values of an arr are in another array. Implementation of torch.isin to support pre 0.10 version.

    Parameters
    ----------
    arr
        The torch tensor to check for availabilities
    values
        The values to search the tensor for.

    Returns
    -------
    a bool tensor of the same shape as :param:`arr` indicating for each
    position whether the element of the tensor is in :param:`values`

    """
    return (arr[..., None] == arr.new(values)).any(-1)
