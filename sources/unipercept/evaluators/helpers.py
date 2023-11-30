"""
General helper functions for evaluators.
"""

import typing as T
import torch
import torch.nn as nn
import torch.special

__all__ = ["DTYPE_INT", "DTYPE_FLOAT", "EPS", "stable_divide"]

DTYPE_INT: T.Final = torch.int64
DTYPE_FLOAT: T.Final = torch.float64
EPS: T.Final[float] = torch.finfo(DTYPE_FLOAT).eps

def stable_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Divide two floating point numbers, but avoid division by zero by adding a small epsilon value.
    """

    return a.to(dtype=DTYPE_FLOAT) / (b.to(dtype=DTYPE_FLOAT) + EPS)
