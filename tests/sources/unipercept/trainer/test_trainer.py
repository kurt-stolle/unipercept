from __future__ import annotations

import pytest

import unipercept as up


@pytest.fixture()
def train_config(tmpdir):
    return up.trainer.TrainConfig(project_name="test", root=str(tmpdir))


def test_trainer(model_factory, train_config, loader_factory):
    callbacks = [
        up.trainer.callbacks.FlowCallback(),
        up.trainer.callbacks.Logger(),
    ]
    trainer = up.trainer.Trainer(
        config=train_config,
        scheduler=up.trainer.SchedulerFactory("poly"),
        optimizer=up.trainer.OptimizerFactory("sgd", lr=0.1),
        callbacks=callbacks,
    )
    trainer.train(model_factory, loader_factory, None)
