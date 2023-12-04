"""
Defines templates for (partial) configuration files.
"""
from __future__ import annotations

import typing as T
from dataclasses import dataclass, is_dataclass

from omegaconf import OmegaConf
from pyexpat import model

from ._lazy import LazyObject

__all__ = [
    "ConfigTemplate",
    "LazyConfigFile",
    "LazyConfigDataPartialFile",
    "LazyConfigEnginePartialFile",
    "LazyConfigModelPartialFile",
]

# ----------------------------- #
# Config templates base classes #
# ----------------------------- #


class ConfigTemplate:
    @classmethod
    def structured(cls) -> T.Self:
        return OmegaConf.structured(cls)  # type: ignore

    def __init__(self, **kwargs):
        if not is_dataclass(self):
            raise TypeError(f"ConfigTemplate {self} is not a dataclass!")


# ----------------------------- #
# Canonical config template     #
# ----------------------------- #

_D = T.TypeVar("_D")
_T = T.TypeVar("_T")
_M = T.TypeVar("_M")


@dataclass
class LazyConfigFile(ConfigTemplate, T.Generic[_D, _T, _M]):
    data: LazyObject[_D]
    engine: LazyObject[_T]
    model: LazyObject[_M]


@dataclass
class LazyConfigDataPartialFile(ConfigTemplate, T.Generic[_D]):
    data: LazyObject[_D]


@dataclass
class LazyConfigEnginePartialFile(ConfigTemplate, T.Generic[_T]):
    engine: LazyObject[_T]


@dataclass
class LazyConfigModelPartialFile(ConfigTemplate, T.Generic[_M]):
    model: LazyObject[_M]
