"""
Impements HP search and NAS using a backend provider.
"""


from __future__ import annotations

import abc
import dataclasses as D
import enum as E
import typing as T

import typing_extensions as TX

from unipercept.utils.typings import Pathable

if T.TYPE_CHECKING:
    pass

_W_contra = T.TypeVar("_W_contra", contravariant=True)

__all__ = ["SearchBackend", "Trial", "TrialWithParameters"]


ConfigOverrides: T.TypeAlias = dict[str, int | float | str | bool]


class SearchBackend(E.StrEnum):
    OPTUNA = E.auto()
    RAY = E.auto()


class Trial(metaclass=abc.ABCMeta):
    __slots__ = ()

    @property
    @abc.abstractmethod
    def config(self) -> ConfigOverrides:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def weights(self) -> Pathable | None:
        raise NotImplementedError


class TrialWithParameters(Trial):
    __slots__ = ("_name", "_config", "_base", "_weights")

    def __init__(
        self,
        name: str,
        config: ConfigOverrides,
        weights: Pathable | None = None,
        parent: Trial | None = None,
    ) -> None:
        self._base = parent
        self._name = name
        self._config = config
        self._weights = weights

    @property
    @TX.override
    def name(self) -> str:
        if self._base is not None:
            return f"{self._base.name}/{self._name}"
        return self._name

    @property
    @TX.override
    def config(self) -> ConfigOverrides:
        if self._base is not None:
            return {**self._base.config, **self._config}
        return self._config

    @property
    @TX.override
    def weights(self) -> Pathable | None:
        if self._weights is not None:
            return self._weights
        if self._base is not None:
            return self._base.weights
        return None


@D.dataclass(slots=True)
class MockTrial:
    name: str
    config: ConfigOverrides = D.field(default_factory=dict)
