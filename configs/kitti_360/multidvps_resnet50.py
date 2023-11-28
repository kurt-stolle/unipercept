from __future__ import annotations

import typing as T

from detectron2.layers import ShapeSpec
from torch import nn
from unimodels import multidvps

import unipercept as up
from unipercept.utils.config import bind as B
from unipercept.utils.config import call as L
from unipercept.utils.config import (
    get_project_name,
    get_session_name,
    use_activation,
    use_norm,
)

from ..cityscapes.multidvps_resnet50 import model, trainer
from ._dataset import DATASET_NAME, data

__all__ = ["model", "data", "trainer"]

trainer.config.project_name="multidvps"
trainer.config.session_name=get_session_name(__file__)
trainer.config.train_batch_size=4
trainer.config.infer_batch_size=4