"""
Defines structures used in the pipeline, usually as a Tensorclass.
"""

from __future__ import annotations

import typing as T

import unipercept.data.points as data_points
from tensordict import TensorDict, TensorDictBase
from torch import Size, Tensor
from unicore.utils.tensorclass import Tensorclass

from ..modules import DepthPrediction, Detection
from ..modules.supervision import Truths

__all__ = ["ThingInstances", "StuffInstances", "PanopticMap", "Context"]


class ThingInstances(Tensorclass):
    kernels: TensorDict
    masks: Tensor
    categories: Tensor
    scores: Tensor
    depths: T.Optional[DepthPrediction]  # (N, ...)
    iids: Tensor  # (N,)

    @property
    def num_instances(self) -> int:
        return self.batch_size[-1]


class StuffInstances(Tensorclass):
    kernels: TensorDict
    masks: Tensor
    categories: Tensor
    scores: Tensor
    depths: T.Optional[DepthPrediction]

    @property
    def num_instances(self) -> int:
        return self.batch_size[-1]


class PanopticMap(Tensorclass):
    semantic: Tensor
    instance: Tensor
    depth: Tensor


class Labels(Tensorclass):
    panseg: Truths
    depth: Tensor | None


class Context(Tensorclass):
    detections: TensorDictBase
    embeddings: TensorDictBase


class SampleID(T.NamedTuple):
    image: int
    sequence: int | None
    frame: int | None


class SampleShape(T.NamedTuple):
    height: int
    width: int


class SampleInfo(T.NamedTuple):
    id: SampleID
    size: SampleShape
    size: SampleShape
    size: SampleShape
