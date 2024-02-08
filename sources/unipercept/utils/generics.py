from __future__ import annotations

import functools
import types
from typing import Final, Generic, TypeVar, cast

from unipercept.utils.missing import MissingValue

_MARKER: Final = "__" + "_".join(n for n in __name__.split(".") if n) + "__"

_T = TypeVar("_T", bound=type)
_NA = MissingValue("NA")


def genericmeta(mcls: _T) -> _T:
    """
    Baseclass for metaclasses that inherit Generic.
    """

    mcls_new = mcls.__new__

    @functools.wraps(mcls_new)
    def wrapper_new(cls, *args, **kwds):
        name: str = kwds.pop("name", _NA)
        bases: tuple[type, ...] = kwds.pop("bases", _NA)
        if _NA.is_missing(bases):
            name, bases, *args = args
        elif _NA.is_missing(name):
            name, *args = args

        if all(not issubclass(b, Generic) or b is Generic for b in bases):
            raise TypeError(
                f"Cannot use {cls.__name__!r} to create {name!r}, none of bases={bases!r} is {Generic!r}"
            )

        bases = types.resolve_bases(bases)
        return mcls_new(cls, name, bases, **kwds)  # type: ignore

    mcls.__new__ = wrapper_new

    return cast(_T, mcls)
