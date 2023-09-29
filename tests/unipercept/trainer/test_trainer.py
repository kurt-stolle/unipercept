import pytest
from unipercept.trainer import (
    OptimizerFactory,
    SchedulerFactory,
    TrainConfig,
    Trainer,
)


@pytest.fixture()
def train_config(tmpdir):
    return TrainConfig(name="test", root=str(tmpdir))


def test_trainer(model, train_config):
    trainer = Trainer(
        model, train_config, scheduler=SchedulerFactory("poly"), optimizer=OptimizerFactory("sgd", lr=0.1)
    )
