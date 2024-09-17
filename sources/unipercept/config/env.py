r"""
Working with environment variables.
"""

from __future__ import annotations

import enum as E
import functools
import os
import typing as T

from distutils.util import strtobool

from unipercept.log import logger

__all__ = ["get_env", "EnvFilter"]

####################
# Environment vars #
####################

_EnvVarT = T.TypeVar("_EnvVarT", int, str, bool)


class EnvFilter(E.StrEnum):
    STRING = E.auto()
    TRUTHY = E.auto()
    FALSY = E.auto()
    POSITIVE = E.auto()
    NEGATIVE = E.auto()
    NONNEGATIVE = E.auto()
    NONPOSITIVE = E.auto()

    @staticmethod
    def apply(f: EnvFilter | str, v: T.Any, /) -> bool:
        if v is None:
            return False
        match EnvFilter(f):
            case EnvFilter.STRING:
                assert isinstance(v, str)
                v = v.lower()
                return v != ""
            case EnvFilter.TRUTHY:
                return bool(v)
            case EnvFilter.FALSY:
                return not bool(v)
            case EnvFilter.POSITIVE:
                return v > 0
            case EnvFilter.NEGATIVE:
                return v < 0
            case EnvFilter.NONNEGATIVE:
                return v >= 0
            case EnvFilter.NONPOSITIVE:
                return v <= 0
            case _:
                msg = f"Invalid filter: {f!r}"
                raise ValueError(msg)


@T.overload
def get_env(
    __type: type[_EnvVarT],
    /,
    *keys: str,
    default: _EnvVarT,
    filter: EnvFilter = EnvFilter.TRUTHY,
) -> _EnvVarT: ...


@T.overload
def get_env(
    __type: type[_EnvVarT],
    /,
    *keys: str,
    default: _EnvVarT | None = None,
    filter: EnvFilter = EnvFilter.TRUTHY,
) -> _EnvVarT | None: ...


@functools.cache
def get_env(
    __type: type[_EnvVarT],
    /,
    *keys: str,
    default: _EnvVarT | None = None,
    filter: EnvFilter = EnvFilter.TRUTHY,
) -> _EnvVarT | None:
    """
    Read an environment variable. If the variable is not set, return the default value.

    If no default is given, an error is raised if the variable is not set.
    """
    keys_read = []
    for k in keys:
        keys_read.append(k)
        v = os.getenv(k)
        if v is None:
            continue
        if __type is bool:
            v = bool(strtobool(v))
        else:
            v = __type(v)
        if not EnvFilter.apply(filter, v):
            continue
        logger.debug("%s = %s (user)", " | ".join(keys_read), str(v))
        break
    else:
        logger.debug("%s = %s (default)", " | ".join(keys), str(default))
        v = default
    return T.cast(_EnvVarT, v)
