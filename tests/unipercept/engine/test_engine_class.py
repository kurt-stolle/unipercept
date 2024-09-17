from __future__ import annotations

import pytest
import unipercept as up
from unipercept.engine._engine import _cleanup_generated_items


@pytest.fixture()
def engine_config():
    return up.engine.EngineParams(project_name="tests")


@pytest.fixture()
def engine_callbacks():
    return [
        up.engine.callbacks.FlowCallback(),
        up.engine.callbacks.Logger(),
    ]


@pytest.fixture()
def scheduler():
    return up.engine.SchedulerFactory("poly", warmup_epochs=1)


@pytest.fixture()
def optimizer():
    return up.engine.OptimizerFactory("sgd", lr=0.1)


def test_engine(model_factory, train_config, loader_factory):
    engine = up.engine.Engine(
        params=train_config,
    )


def test_cleanup_generated_items(tmp_path):
    # Create some files in the temporary directory
    files = ["item-1", "item-600", "otherkey-200", "last-800", "item-123"]
    for file in files:
        (tmp_path / file).touch()

    # Call the function
    _cleanup_generated_items(tmp_path, 3)

    # Check that only the expected files remain
    remaining_files = list(tmp_path.iterdir())
    assert len(remaining_files) == 3
    assert tmp_path / "otherkey-200" in remaining_files
    assert tmp_path / "item-600" in remaining_files
    assert tmp_path / "last-800" in remaining_files


def test_multi_stage_training():
    pass  # TODO implement test
