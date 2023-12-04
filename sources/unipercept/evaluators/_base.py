"""
Implements the baseclass for evaluators
"""
from __future__ import annotations
import dataclasses as D
import abc
from multiprocessing import Process
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


class EvaluatorComputeKWArgs(T.Protocol):
    """
    A type for the keyword arguments passed to the ``compute`` method of an evaluator.
    """

    device: torch.types.Device
    path: str


@D.dataclass(kw_only=True)
class Evaluator(T.Protocol, metaclass=abc.ABCMeta):
    """
    Implements a stateless evaluator for a given task.
    """

    def update(self, storage: TensorDictBase, outputs: ModelOutput) -> None:
        return None

    @abc.abstractmethod
    def compute(
        self, storage: TensorDictBase, **kwargs: EvaluatorComputeKWArgs
    ) -> dict[str, int | float | str | bool | dict]:
        return {}

    @abc.abstractmethod
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        return {}
