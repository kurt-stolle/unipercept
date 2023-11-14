from __future__ import annotations

import typing as T
from logging import warn
from pathlib import Path

from detectron2.layers import ShapeSpec
from torch import nn
from unimodels import risknet

import unipercept as up
from unipercept.utils.config import get_project_name, get_session_name
from unipercept.utils.config._lazy import bind as B
from unipercept.utils.config._lazy import call as L
from unipercept.utils.config._lazy import use_activation, use_norm

from .data._risk import data

__all__ = ["model", "data", "trainer"]

_INFO = up.data.read_info(data)

trainer = B(up.trainer.Trainer)(
    config=L(up.trainer.config.TrainConfig)(
        project_name=get_project_name(__file__),
        session_name=get_session_name(),
        train_epochs=50,
        eval_epochs=1,
        save_epochs=1,
    ),
    optimizer=L(up.trainer._optimizer.OptimizerFactory)(
        opt="adamw",
    ),
    scheduler=L(up.trainer._scheduler.SchedulerFactory)(
        scd="poly",
        warmup_epochs=1,
    ),
    callbacks=[up.trainer.callbacks.FlowCallback, up.trainer.callbacks.ProgressCallback],
)

model = B(risknet.RiskNet)(
    backbone=L(up.nn.backbones.timm.TimmBackbone)(name="resnet50"),
    example_layer=L(nn.Linear)(
        in_features=2048,
        out_features=1,
    ),
    example_loss=L(nn.Identity)(),
)
