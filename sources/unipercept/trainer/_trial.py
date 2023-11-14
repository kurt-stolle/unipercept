"""
Impements HP search and NAS using a backend provider.
"""


from __future__ import annotations

import abc
import dataclasses as D
import enum as E
import typing as T

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing_extensions import override

from unipercept.utils.config import LazyObject, instantiate

if T.TYPE_CHECKING:
    import unipercept as up

_W_contra = T.TypeVar("_W_contra", contravariant=True)

__all__ = ["SearchBackend", "Trial"]


class SearchBackend(E.StrEnum):
    OPTUNA = E.auto()
    RAY = E.auto()


@abc.abstractmethod
class Trial:
    @property
    @abc.abstractmethod
    def params(self) -> T.Dict[str, T.Any]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError


@D.dataclass(slots=True, frozen=True)
class MockTrial:
    name: str
    params: dict[str, T.Any] = D.field(default_factory=dict)
