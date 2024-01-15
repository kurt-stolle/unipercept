import copy
import re
from types import MappingProxyType
from typing import ClassVar, Mapping, Optional, Sequence

from typing_extensions import Self, override


class Matchable:
    __match_groups: ClassVar[Optional[Sequence[str | int]]] = None
    __match_kwgroups: ClassVar[Optional[Mapping[str, str | int]]] = None

    def __new__(cls, *args, **kwargs):
        if cls is Matchable:
            raise TypeError(f"Cannot instantiate {cls.__name__} directly")
        return super().__new__(cls)

    @override
    def __init_subclass__(
        cls,
        /,
        match_groups: Optional[Sequence[str | int]] = None,
        match_kwgroups: Optional[Mapping[str, str | int]] = None,
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
