from __future__ import annotations

import typing as T
from logging import warn

import unipercept as up
from unipercept.config import get_project_name, get_session_name
from unipercept.config._lazy import bind as B
from unipercept.config._lazy import call as L

from ..cityscapes.multidvps_resnet50 import model
from .data._dvps import data

__all__ = ["model", "data", "engine"]

engine = B(up.engine.Engine)(
    config=L(up.engine.params.EngineParams)(
        project_name=get_project_name(__file__),
        session_name=get_session_name(),
        # train_steps=120_000,
        train_epochs=20,
        eval_epochs=1,
        save_epochs=1,
    ),
    optimizer=L(up.engine._optimizer.OptimizerFactory)(
        opt="adamw",
    ),
    scheduler=L(up.engine._scheduler.SchedulerFactory)(
        scd="poly",
        warmup_epochs=1,
    ),
    callbacks=[up.engine.callbacks.FlowCallback, up.engine.callbacks.ProgressCallback],
)
