"""
Implementation of metrics to evaluate object tracking performance.
"""

from __future__ import annotations

import dataclasses as D
import typing as T

import torch
import torch.types
from tensordict import TensorDict, TensorDictBase

from ..model import ModelOutput
from ._base import Evaluator

TRUE_OBJECTS = "true_objects"
PRED_OBJECTS = "pred_objects"


class ObjectTrackingEvaluator(Evaluator):
    """
    Implements HOTA/MOTA and friends for object tracking.
    """

    @classmethod
    def from_info(cls, name: str) -> T.Self:
        return cls()

    def update(self, storage: TensorDictBase, outputs: ModelOutput) -> TensorDict | None:
        raise NotImplementedError

    def compute(self, storage: TensorDictBase, *, device: torch.types.Device) -> dict[str, int | float | str | bool]:
        raise NotImplementedError
