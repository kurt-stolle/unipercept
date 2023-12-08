"""
Weights & Biases integration.

See: https://wandb.ai
"""

from __future__ import annotations

import enum as E
import os
import tempfile

import torch.nn as nn
import typing_extensions as TX
import wandb

from unipercept import read_config
from unipercept.engine import EngineParams
from unipercept.engine.callbacks import CallbackDispatcher, Signal, State
from unipercept.log import get_logger
from unipercept.state import check_main_process, on_main_process
from unipercept.utils.time import ProfileAccumulator

__all__ = ["WandBCallback", "pull_config", "pull_engine", "ArtifactType"]

_logger = get_logger(__name__)


class ArtifactType(E.StrEnum):
    RUN = E.auto()
    CONFIG = E.auto()
    MODEL = E.auto()
    STATE = E.auto()


def pull_config(name: str):
    """
    Pulls a configuration file from a WandB artifact.
    """

    artifact = wandb.Api().artifact(name, type=ArtifactType.CONFIG.value)
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact.checkout(root=temp_dir)
        config = read_config(os.path.join(temp_dir, "config.yaml"))
    return config


def pull_engine(name: str):
    """
    Pulls the engine state from a WandB artifact.
    """
    pass


class WandBCallback(CallbackDispatcher):
    """
    Extended integration with Weights & Biases.

    Since Accelerate already provides a WandB integration, this callback only adds new features.
    """

    @TX.override
    @on_main_process()
    def on_trackers_setup(self, params: EngineParams, state: State, control: Signal, *, model: nn.Module):
        run = wandb.run
        assert run is not None, "WandB run not initialized"

        _logger.info(f"Logging additional metrics to WandB run {run.name}")

        run.watch(model, log="all", log_freq=params.logging_steps)
        run.log_code("./sources", include_fn=lambda path, root: path.endswith(".py"))

    @TX.override
    @on_main_process()
    def on_save(
        self, params: EngineParams, state: State, control: Signal, *, model_path: str, state_path: str, **kwargs
    ):
        run = wandb.run
        assert run is not None, "WandB run not initialized"
        run_name = run.name
        assert run_name is not None, "WandB run name not initialized"

        try:
            _logger.info(f"Logging model to WandB run {run.name}")
            run.log_model(model_path, name=f"model-{run_name}")
        except Exception as err:
            _logger.warning(f"Failed to log model to WandB run {run.name}: {err}")

        try:
            _logger.info(f"Logging engine state to WandB run {run.name}")
            state_artifact = wandb.Artifact(
                name=f"state-{run_name}",
                type=ArtifactType.STATE.value,
                description="Engine state",
            )
            state_artifact.add_dir(state_path)

            wandb.log_artifact(
                state_artifact,
                type=ArtifactType.STATE.value,
                description="Engine state",
            )
        except Exception as err:
            _logger.warning(f"Failed to log model to WandB run {run.name}: {err}")

    @TX.override
    @on_main_process()
    def on_inference_end(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        timings: ProfileAccumulator,
        results_path: str,
        **kwargs,
    ):
        run = wandb.run
        assert run is not None, "WandB run not initialized"
        run_name = run.name
        assert run_name is not None

        _logger.info("Logging evaluation outcomes to WandB")
        run.log_artifact(
            results_path,
            name=f"inference-{run_name}",
            type=ArtifactType.RUN.value,
        )
        run.log({"inference/timings": wandb.Table(dataframe=timings.to_summary())}, commit=False)
