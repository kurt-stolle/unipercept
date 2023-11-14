from __future__ import annotations

import typing as T
from logging import warn

import unipercept as up
from unipercept.utils.config import get_project_name, get_session_name
from unipercept.utils.config._lazy import bind as B
from unipercept.utils.config._lazy import call as L

from ..cityscapes.multidvps_resnet50 import model
from .data._dvps import data

__all__ = ["model", "data", "trainer"]

trainer = B(up.trainer.Trainer)(
    config=L(up.trainer.config.TrainConfig)(
        project_name=get_project_name(__file__),
        session_name=get_session_name(),
        # train_steps=120_000,
        train_epochs=20,
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
