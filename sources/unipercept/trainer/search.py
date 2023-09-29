"""
Impements HP search and NAS using a backend provider.
"""


from __future__ import annotations

import dataclasses
import enum
import typing as T

if T.TYPE_CHECKING:
    import optuna
    import ray.tune

    _W_contra = T.TypeVar("_W_contra", optuna.Trial, dict, contravariant=True)
else:
    _W_contra = T.TypeVar("_W_contra", contravariant=True)

__all__ = ["SearchBackend", "Trial"]


class SearchBackend(enum.StrEnum):
    OPTUNA = enum.auto()
    RAY = enum.auto()


@dataclasses.dataclass(slots=True)
class Trial(T.Generic[_W_contra]):
    wrap: _W_contra

    @property
    def name(self):
        # TODO: rewrite
        raise NotImplementedError

    @property
    def params(self):
        # TODO: rewrite
        raise NotImplementedError
