from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import ClassVar, Self, override


class Matchable:
    __match_groups: ClassVar[Sequence[str | int] | None] = None
    __match_kwgroups: ClassVar[Mapping[str, str | int] | None] = None

    def __new__(cls, *args, **kwargs):
        if cls is Matchable:
            raise TypeError(f"Cannot instantiate {cls.__name__} directly")
        return super().__new__(cls)

    @override
    def __init_subclass__(
        cls,
        /,
        match_groups: Sequence[str | int] | None = None,
        match_kwgroups: Mapping[str, str | int] | None = None,
    ) -> None:
        if match_groups is not None:
            cls.__match_groups = tuple(match_groups)
        if match_kwgroups is not None:
            cls.__match_kwgroups = MappingProxyType(copy.copy(match_kwgroups))

    @classmethod
    def from_match(cls, match: re.Match) -> Self:
        if cls.__match_groups is not None:
            args = (match.group(g) for g in cls.__match_groups)
        else:
            args = ()
        if cls.__match_kwgroups is not None:
            kwargs = {k: match.group(g) for k, g in cls.__match_kwgroups.items()}
        else:
            kwargs = {}

        return cls(*args, **kwargs)
