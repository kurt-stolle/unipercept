from __future__ import annotations

import pytest
from unipercept.engine import EngineParams


@pytest.mark.parametrize(
    ("attr_step, attr_epoch, fn_to_step, fn_to_epoch"),
    [
        ("train_steps", "train_epochs", "get_train_steps", "get_train_epochs"),
        (
            "eval_steps",
            "eval_epochs",
            "get_eval_interval_steps",
            "get_eval_interval_epochs",
        ),
        (
            "save_steps",
            "save_epochs",
            "get_save_interval_steps",
            "get_save_interval_epochs",
        ),
    ],
)
def test_epochs_step(tmp_path, attr_step, attr_epoch, fn_to_step, fn_to_epoch):
    epochs = 6
    steps_per_epoch = 9
    steps = epochs * steps_per_epoch

    cfg_epochs = EngineParams(
        project_name="testing",
        session_name="testing",
        root=str(tmp_path),
        **{attr_epoch: epochs},
    )  # type: ignore
    assert cfg_epochs.__getattribute__(attr_epoch) == epochs

    epochs2epochs = cfg_epochs.__getattribute__(fn_to_epoch)(steps_per_epoch)
    assert epochs2epochs == epochs

    cfg_steps = EngineParams(
        project_name="testing",
        session_name="testing",
        root=str(tmp_path),
        **{attr_step: steps},
    )  # type: ignore
    assert cfg_steps.__getattribute__(attr_step) == steps

    steps2steps = cfg_steps.__getattribute__(fn_to_step)(steps_per_epoch)
    assert steps2steps == steps

    epochs2steps = cfg_epochs.__getattribute__(fn_to_step)(steps_per_epoch)
    assert epochs2steps == steps2steps

    steps2epochs = cfg_steps.__getattribute__(fn_to_epoch)(steps_per_epoch)
    assert steps2epochs == epochs2epochs
