from __future__ import annotations

import pytest

import unipercept as up


@pytest.fixture()
def train_config(tmpdir):
    return up.engine.EngineParams(project_name="test", root=str(tmpdir))


def test_engine(model_factory, train_config, loader_factory):
    callbacks = [
        up.engine.callbacks.FlowCallback(),
        up.engine.callbacks.Logger(),
    ]
    engine = up.engine.Engine(
        params=train_config,
        scheduler=up.engine.SchedulerFactory("poly"),
        optimizer=up.engine.OptimizerFactory("sgd", lr=0.1),
        callbacks=callbacks,
    )
    engine.train(model_factory, loader_factory, None)
