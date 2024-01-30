"""
Impements HP search and NAS using a backend provider.
"""


from __future__ import annotations

import abc
import dataclasses as D
import enum as E
import typing as T



if T.TYPE_CHECKING:
    pass

_W_contra = T.TypeVar("_W_contra", contravariant=True)

__all__ = ["SearchBackend", "Trial"]


SearchParams: T.TypeAlias = dict[str, int | float | str | bool]


class SearchBackend(E.StrEnum):
    OPTUNA = E.auto()
    RAY = E.auto()


class Trial(metaclass=abc.ABCMeta):
    __slots__ = ()

    @property
    @abc.abstractmethod
    def params(self) -> SearchParams:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError


@D.dataclass(slots=True)
class MockTrial:
    name: str
    params: SearchParams = D.field(default_factory=dict)
