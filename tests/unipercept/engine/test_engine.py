from __future__ import annotations

import typing as T

import pytest
import typing_extensions as TX

import unipercept as up
from unipercept.engine._engine import _cleanup_generated_items


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
