"""
Defines structures used in the pipeline, usually as a Tensorclass.
"""

from __future__ import annotations

import typing as T

from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from unicore.utils.tensorclass import Tensorclass

import unipercept as up

from ..modules import DepthPrediction

__all__ = ["ThingInstances", "StuffInstances", "Context"]


class ThingInstances(Tensorclass):
    kernels: TensorDict
    logits: Tensor
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
    logits: Tensor
    categories: Tensor
    scores: Tensor
    depths: T.Optional[DepthPrediction]

    @property
    def num_instances(self) -> int:
        return self.batch_size[-1]


class Context(Tensorclass):
    """
    Context object holds all the data that is needed for the pipeline to run.
    """

    captures: up.model.CaptureData
    detections: TensorDictBase
    embeddings: TensorDictBase

    @property
    def input_size(self) -> T.Sequence[int]:
        return self.captures.images.shape[-2:]
