"""
Implements the baseclass for evaluators
"""
from __future__ import annotations

import abc
import dataclasses as D
import enum
import typing as T

import torch
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDict, TensorDictBase

if T.TYPE_CHECKING:
    from ..model import InputData, ModelOutput
    from ..data import DataLoaderFactory

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
class Evaluator(metaclass=abc.ABCMeta):
    """
    Implements a stateless evaluator for a given task.
    """

    def update(self, storage: TensorDictBase, inputs: InputData, outputs: ModelOutput) -> None:
        return None

    @abc.abstractmethod
    def compute(
        self, storage: TensorDictBase, **kwargs: EvaluatorComputeKWArgs
    ) -> dict[str, int | float | str | bool | dict]:
        return {}

    @abc.abstractmethod
    def plot(self, storage: TensorDictBase) -> dict[str, pil_image.Image]:
        return {}
