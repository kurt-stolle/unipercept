"""
Environment variables configuration module.
"""

from __future__ import annotations

import enum
import os
import typing as T
from distutils.util import strtobool
from pathlib import Path

__all__ = ["get_env", "CONFIG_ROOT"]


CONFIG_ROOT = os.getenv("UNI_CONFIGS", str(Path("./configs").resolve()))

_R = T.TypeVar("_R", int, str, bool)


class EnvFilter(enum.StrEnum):
    STRING = enum.auto()
    TRUTHY = enum.auto()
    FALSY = enum.auto()
    POSITIVE = enum.auto()
    NEGATIVE = enum.auto()
    NONNEGATIVE = enum.auto()
    NONPOSITIVE = enum.auto()

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
                raise ValueError(f"Invalid filter: {f!r}")


@T.overload
def get_env(__type: type[_R], /, *keys: str, default: _R, filter: EnvFilter = EnvFilter.TRUTHY) -> _R:
    ...


@T.overload
def get_env(
    __type: type[_R], /, *keys: str, default: _R | None = None, filter: EnvFilter = EnvFilter.TRUTHY
) -> _R | None:
    ...


def get_env(
    __type: type[_R], /, *keys: str, default: _R | None = None, filter: EnvFilter = EnvFilter.TRUTHY
) -> _R | None:
    """
    Read an environment variable. If the variable is not set, return the default value.

    If no default is given, an error is raised if the variable is not set.
    """
    for k in keys:
        v = os.getenv(k)
        if v is None:
            continue
        if __type is bool:
            v = bool(strtobool(v))
        else:
            v = __type(v)
        if not EnvFilter.apply(filter, v):
            continue
        break
    else:
        v = default
    return T.cast(_R, v)
