"""
Weights & Biases integration.

See: https://wandb.ai
"""

from __future__ import annotations

import enum as E
import os
import tempfile
import dataclasses as D
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
    INFERENCE = E.auto()


class WandBWatchMode(E.StrEnum):
    """
    Watch mode for Weights & Biases.
    """

    GRADIENTS = E.auto()
    PARAMETERS = E.auto()
    ALL = E.auto()


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


##################
# WANDB CALLBACK #
##################


@D.dataclass(slots=True)
class WandBCallback(CallbackDispatcher):
    """
    Extended integration with Weights & Biases.

    Since Accelerate already provides a WandB integration, this callback only adds new features.
    """

    watch_model: WandBWatchMode = WandBWatchMode.GRADIENTS
    upload_code: bool = True
    model_history: int = 1
    state_history: int = 1
    inference_history: int = 0
    tabulate_inference_timings: bool = True

    @TX.override
    @on_main_process()
    def on_trackers_setup(self, params: EngineParams, state: State, control: Signal, *, model: nn.Module):
        if wandb.run is None:
            return

        run = wandb.run

        _logger.info(f"Logging additional metrics to WandB run {run.name}")

        if self.watch_model is not None:
            run.watch(model, log=self.watch_model.value, log_freq=params.logging_steps)
        if self.upload_code:
            run.log_code("./sources", include_fn=lambda path, root: path.endswith(".py"))

    @TX.override
    @on_main_process()
    def on_save(
        self, params: EngineParams, state: State, control: Signal, *, model_path: str, state_path: str, **kwargs
    ):
        if wandb.run is None:
            return
        if self.model_history > 0:
            self._log_model(model_path)
        if self.state_history > 0:
            self._log_state(state_path)

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
        if wandb.run is None:
            return
        if self.inference_history > 0:
            self._log_inference(results_path)
        if self.tabulate_inference_timings:
            self._log_profiling("inference/timings", timings)

    def _log_model(self, model_path: str):
        run = wandb.run
        assert run is not None, "WandB run not initialized"

        try:
            _logger.info(f"Logging model to WandB run {run.name}")
            name = f"model-{run.name}"
            run.log_model(model_path, name=f"model-{run.name}")

            artifact = wandb.Api().artifact(
                f"{run.entity}/{run.project_name()}/{name}:latest", type=ArtifactType.MODEL.value
            )
            artifact_historic_delete(artifact, self.model_history)
        except Exception as err:
            _logger.warning(f"Failed to log model to WandB run {run.name}: {err}")

    def _log_state(self, state_path: str):
        run = wandb.run
        assert run is not None, "WandB run not initialized"
        try:
            _logger.info(f"Logging engine state to WandB run {run.name}")
            state_artifact = wandb.Artifact(
                name=f"state-{run.name}",
                type=ArtifactType.STATE.value,
            )
            state_artifact.add_dir(state_path)

            artifact = wandb.log_artifact(
                state_artifact,
                type=ArtifactType.STATE.value,
            )
            artifact.wait()

            artifact_historic_delete(artifact, self.state_history)
        except Exception as err:
            _logger.warning(f"Failed to log model to WandB run {run.name}: {err}")

    def _log_profiling(self, key: str, timings: ProfileAccumulator):
        run = wandb.run
        assert run is not None, "WandB run not initialized"
        run.log({key: wandb.Table(dataframe=timings.to_summary())}, commit=False)

    def _log_inference(self, results_path: str):
        run = wandb.run
        assert run is not None, "WandB run not initialized"
        try:
            _logger.info("Logging evaluation outcomes to WandB")
            artifact = run.log_artifact(
                results_path,
                name=f"inference-{run.name}",
                type=ArtifactType.INFERENCE.value,
            )
            artifact.wait()

            artifact_historic_delete(artifact, self.inference_history)
        except Exception as err:
            _logger.warning(f"Failed to log inference to WandB run {run.name}: {err}")


#######################
# ARTIFACT MANAGEMENT #
#######################

SPLIT_QUALIFIER = "/"
SPLIT_VERSION = ":"


def artifact_name_is_qualified(str) -> bool:
    """
    Check whether a string is a qualified name.
    """
    return SPLIT_QUALIFIER in str


def artifact_name_has_version(str) -> bool:
    """
    Check whether a string is a qualified name.
    """
    return SPLIT_VERSION in str


def artifact_version_as_int(artifact: wandb.Artifact) -> int:
    """
    Get the version of an artifact as an integer.
    """
    ver_str = artifact.version[1:]

    return int(ver_str)


def artifact_historic_delete(artifact: wandb.Artifact, keep: int) -> None:
    """
    Delete historic artifacts from a run, useful to save space.

    Parameters
    ----------
    name
        Name of the run.
    keep
        Number of artifacts to keep.
    """

    name, *_ = artifact.qualified_name.split(SPLIT_VERSION, 1)

    api = wandb.Api()

    vs = api.artifact_versions(type_name=artifact.type, name=name)
    vs = sorted(vs, key=artifact_version_as_int, reverse=True)
    for artifact in vs[keep:]:
        _logger.info(f"Deleting artifact {name} version {artifact.version}")
        artifact.delete(delete_aliases=True)
