"""
Implements the baseclass for evaluators
"""
from __future__ import annotations
import dataclasses as D
import abc
import typing as T

import torch
import torch.types
import enum

from PIL import Image as pil_image
from tensordict import TensorDict, TensorDictBase

if T.TYPE_CHECKING:
    from ..model import ModelOutput

__all__ = ["Evaluator", "PlotMode"]


class PlotMode(enum.Enum):
    ALWAYS = enum.auto()
    ONCE = enum.auto()
    NEVER = enum.auto()


@D.dataclass(kw_only=True)
class Evaluator(T.Protocol, metaclass=abc.ABCMeta):
    """
    Implements a stateless evaluator for a given task.
    """

    @abc.abstractmethod
    def update(self, storage: TensorDictBase, outputs: ModelOutput) -> TensorDict | None:
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self, storage: TensorDictBase, *, device: torch.types.Device) -> dict[str, int | float | str | bool]:
        raise NotImplementedError

    @abc.abstractmethod
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        raise NotImplementedError
