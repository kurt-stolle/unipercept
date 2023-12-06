from __future__ import annotations

import typing as T

from detectron2.layers import ShapeSpec
from torch import nn
from unimodels import multidvps

import unipercept as up
from unipercept.config import bind as B
from unipercept.config import call as L
from unipercept.config import (
    get_project_name,
    get_session_name,
    use_activation,
    use_norm,
)

from ..cityscapes.multidvps_resnet50 import engine, model
from ._dataset import DATASET_NAME, data

__all__ = ["model", "data", "engine"]

engine.params.project_name = "multidvps"
engine.params.session_name = get_session_name(__file__)
engine.params.train_batch_size = 4
engine.params.infer_batch_size = 4
