r"""
Checks for various conditions at runtime. Ignored at compile/fx/export. Similar to
``assert``ions.
"""

from __future__ import annotations

import typing as T
from types import EllipsisType

from torch import Tensor


def assert_shape(
    x: Tensor, shape: T.Iterable[str | int | EllipsisType], *, raises: bool = True
) -> bool:
    r"""
    Check whether a tensor has a specified shape.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `Tensor`.

    Parameters
    ----------
    x: Tensor
        The tensor to check.
    shape: Iterable[str | int | EllipsisType]
        The expected shape of the tensor. The ellipsis can be used to match any
        number of dimensions.
    raises: bool
        Whether to raise an exception if the shape does not match. By default, True.

    Raises
    ------
    TypeError: if the input tensor is not a tensor.

    Examples
    --------
    >>> x = torch.rand(2, 3, 8, 16)
    >>> assert_shape(x, ("B", "C", "H", "W"))
    True

    >>> x = torch.rand(2, 3, 8, 16)
    >>> assert_shape(x, ("B", "C", "H", "H"))
    False

    >>> x = torch.rand(2, 1, 1, 2, 1, 8, 16)
    >>> assert_shape(x, (..., "H", "W"))
    True

    >>> x = torch.rand(2, 3, 8, 16)
    >>> assert_shape(x, (2, 3, "H", "W"))
    True

    >>> x = torch.rand(2, 3, 8, 16)
    >>> assert_shape(x, (2, 3, "H", ...))
    True
    """
    if not assert_tensor(x, raises=raises):
        return False

    tgt = list(shape)
    src = list(x.shape)
    if tgt[0] == ...:
        src = x.shape[-len(tgt) + 1 :]
        tgt.pop(0)
    if tgt[-1] == ...:
        src = x.shape[: len(tgt) - 1]
        tgt.pop(-1)

    if any(dim is ... for dim in tgt):
        msg = f"Ellipsis must be at the beginning or end of the shape. Got {shape}"
        raise ValueError(msg)

    if len(src) != len(tgt):
        if raises:
            msg = f"{x} shape must be [{shape}]. Got {x.shape}"
            raise TypeError(msg)
        return False

    for i in range(len(src)):
        dim: str | int | EllipsisType = tgt[i]
        if isinstance(dim, str):
            continue
        if isinstance(dim, int):
            if src[i] != dim:
                if raises:
                    msg = f"{x} shape must be [{shape}]. Got {x.shape}"
                    raise AssertionError(msg)
                return False
        else:
            msg = (
                f"Invalid shape dimension. Got {dim=} for {shape=}. "
                "Checked dimensions must be a string or integer."
            )
            raise ValueError(msg)

    return True


def assert_tensor(x: object, *, raises: bool = True) -> T.TypeGuard[Tensor]:
    r"""
    Check whether a variable is a tensor.

    Parameters
    ----------
    x: object
        The variable to check.
    raises: bool
        Whether to raise an exception if the variable is not a tensor. By default, True.

    Examples
    --------
    >>> x = torch.rand(2, 3, 8, 16)
    >>> assert_tensor(x)
    True

    >>> x = 5
    >>> assert_tensor(x)
    False
    """
    if not isinstance(x, Tensor):
        if raises:
            msg = f"{x} must be a tensor. Got {type(x)}"
            raise AssertionError(msg)
        return False
    return True
