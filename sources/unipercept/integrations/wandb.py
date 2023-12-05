"""
Weights & Biases integration.

See: https://wandb.ai
"""

from __future__ import annotations
from unipercept.engine import EngineParams

import wandb
import wandb.sdk
import enum as E
import tempfile
import typing as T
import typing_extensions as TX
import torch.nn as nn
import os

from unipercept import read_config
from unipercept.engine.callbacks import CallbackDispatcher, Signal, State
from unipercept.utils.state import check_main_process
from unipercept.log import get_logger
from accelerate import Accelerator


__all__ = ["WandBCallback", "pull_config", "pull_engine", "ArtifactType"]

_logger = get_logger(__name__)


class ArtifactType(E.StrEnum):
    RUN = E.auto()
    CONFIG = E.auto()
    MODEL = E.auto()


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
    def on_trackers_setup(self, params: EngineParams, state: State, control: Signal, *, model: nn.Module):
        if not check_main_process():
            return

        run = wandb.run
        assert run is not None, "WandB run not initialized"

        _logger.info(f"Logging additional metrics to WandB run {run.name}")

        run.watch(model, log="all", log_freq=params.logging_steps)

    @TX.override
    def on_save(
        self, params: EngineParams, state: State, control: Signal, *, model_path: str, state_path: str, **kwargs
    ):
        if not check_main_process():
            return

        run = wandb.run
        assert run is not None, "WandB run not initialized"

        _logger.info(f"Logging model to WandB run {run.name}")

        run.log_model(model_path)
