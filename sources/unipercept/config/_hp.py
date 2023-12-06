"""
Implements configuration directives for working with hyperparameters.
"""

from __future__ import annotations

import typing as T

_T = T.TypeVar("_T", covariant=True)


class HP(T.Generic[_T]):
    """
    Defines a hyperparameter. This is a wrapper around a value that can be used to define a hyperparameter in a
    configuration. The value is not evaluated until the hyperparameter is called. This is useful for defining
    hyperparameters that are not known at configuration time.
    """

    def __init__(self, *args, default: _T):
        self.default = default
        self.options = args

    def __call__(self) -> _T:
        return self.default
