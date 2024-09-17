"""
Numerical stability
-------------------
Some implementations use `torch.finfo(torch.float32).eps` to determine the epsilon
value used to enforce numeric stability. This value is the smallest positive
floating point value that can be represented in the given floating point type.
However, this value is not always the best choice for enforcing numeric stability
in all cases. For example, when comparing two floating point numbers, the difference
between the two numbers should be greater than the epsilon value to be considered
significant. In this case, a value of `1e-6` is used as the default epsilon value
to enforce numeric stability (i.e. the resolution of the floating point numbers).

For exponential functions, the input value is clamped to a maximum
of `88.0` to prevent overflow. This value is chosen because `exp(88.7)` is the
largest value that can be represented in 32-bit floating point precision without
overflowing.

Generally, we follow the following rules for numeric stability:
    - Division by zero is prevented by **adding** a small epsilon value to the denominator.
    - Logarithm of zero is prevented by **adding** a small epsilon value to the input.
    - Square root of negative values is prevented by **clamping** the input to a small value.
    - Exponential functions are **clamped **to a maximum input value to prevent overflow.
"""

import math
from typing import Final

import torch

from unipercept.types import Tensor

__all__ = []  # all symbols are not exported

EPS_FP32: Final = torch.finfo(torch.float32).eps
EPS_DIV_FP32: Final = torch.finfo(torch.float32).eps
MAX_EXP_FP32: Final = math.floor(math.log(torch.finfo(torch.float32).max))
EPS_FP16: Final = torch.finfo(torch.float16).eps
MAX_EXP_FP16: Final = math.floor(math.log(torch.finfo(torch.float16).max))
EPS_BF16: Final = torch.finfo(torch.bfloat16).eps
MAX_EXP_BF16: Final = math.floor(math.log(torch.finfo(torch.bfloat16).max))


def div(a: Tensor, b: Tensor, *, eps: float = EPS_DIV_FP32) -> Tensor:
    return a / (b + eps)


def log(x: Tensor, *, eps: float) -> Tensor:
    return torch.log(x + eps)


def sqrt(x: Tensor, *, eps: float) -> Tensor:
    return x.clamp_min(eps).sqrt()


def exp(x: Tensor, *, max_exp: float) -> Tensor:
    return x.clamp_max(max_exp).exp()
