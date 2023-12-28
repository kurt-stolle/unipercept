"""
Weights & Biases integration.

See: https://wandb.ai
"""

from __future__ import annotations
import functools
import typing as T
import dataclasses as D
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
from unipercept.state import on_main_process
from unipercept.utils.time import ProfileAccumulator

if T.TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandBRun

__all__ = ["WandBCallback", "pull_config", "pull_engine", "ArtifactType", "skip_no_run"]

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


_P = T.ParamSpec("_P")
_R = T.TypeVar("_R", bound=None)


def skip_no_run(fn: T.Callable[_P, _R | None]) -> T.Callable[_P, _R | None]:
    """
    Decorator that skips a function if there is no WandB run.
    """

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
        if wandb.run is None:
            return
        return fn(*args, **kwargs)

    return wrapper


##################
# WANDB CALLBACK #
##################


@D.dataclass(slots=True)
class WandBCallback(CallbackDispatcher):
    """
    Extended integration with Weights & Biases.

    Since Accelerate already provides a WandB integration, this callback only adds new features.
    """

    watch_model: WandBWatchMode | None | str = None
    watch_steps: int | None = D.field(
        default=None,
        metadata={
            "help": (
                "Interval passed to W&B model watcher. "
                "When set to `None`, the 'logging_steps' attribute is taken from the engine parameters."
            )
        },
    )
    upload_code: bool = True
    model_history: int = 1
    state_history: int = 1
    inference_history: int = 0
    tabulate_inference_timings: bool = False

    @property
    def run(self) -> WandBRun:
        """
        Current WandB run.
        """
        run = wandb.run
        if run is None:
            raise RuntimeError("WandB run not initialized")
        return run

    @TX.override
    @skip_no_run
    @on_main_process()
    def on_trackers_setup(self, params: EngineParams, state: State, control: Signal, **kwargs):
        _logger.info(f"Logging additional metrics to WandB run {self.run.name}")
        if self.upload_code:
            self.run.log_code("./sources", include_fn=lambda path, root: path.endswith(".py"))

    @TX.override
    @skip_no_run
    @on_main_process()
    def on_save(
        self, params: EngineParams, state: State, control: Signal, *, model_path: str, state_path: str, **kwargs
    ):
        if self.model_history > 0:
            self._log_model(model_path)
        if self.state_history > 0:
            self._log_state(state_path)

    @TX.override
    @skip_no_run
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
        if self.inference_history > 0:
            self._log_inference(results_path)
        if self.tabulate_inference_timings:
            self._log_profiling("inference/timings", timings)

    @TX.override
    @skip_no_run
    def on_train_begin(self, params: EngineParams, state: State, control: Signal, *, model: nn.Module, **kwargs):
        if self.watch_model is not None:
            self.run.watch(
                model,
                log=self.watch_model.value,
                log_freq=self.watch_steps if self.watch_steps is not None else params.logging_steps,
            )

    def _log_model(self, model_path: str):
        try:
            _logger.info(f"Logging model to WandB run {self.run.name}")
            name = f"model-{self.run.name}"
            self.run.log_model(model_path, name=f"model-{self.run.name}")

            artifact = wandb.Api().artifact(
                f"{self.run.entity}/{self.run.project_name()}/{name}:latest", type=ArtifactType.MODEL.value
            )
            artifact_historic_delete(artifact, self.model_history)
        except Exception as err:
            _logger.warning(f"Failed to log model to WandB run {self.run.name}: {err}")

    def _log_state(self, state_path: str):
        try:
            _logger.info(f"Logging engine state to WandB: {self.run.name}")
            state_artifact = wandb.Artifact(
                name=f"state-{self.run.name}",
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
            _logger.warning(f"Failed to log model to WandB self.run {self.run.name}: {err}")

    def _log_profiling(self, key: str, timings: ProfileAccumulator):
        run = wandb.run
        assert run is not None, "WandB run not initialized"
        run.log({key: wandb.Table(dataframe=timings.to_summary())}, commit=False)

    def _log_inference(self, results_path: str):
        try:
            _logger.info("Logging evaluation outcomes to WandB")
            artifact = self.run.log_artifact(
                results_path,
                name=f"inference-{self.run.name}",
                type=ArtifactType.INFERENCE.value,
            )
            artifact.wait()

            artifact_historic_delete(artifact, self.inference_history)
        except Exception as err:
            _logger.warning(f"Failed to log inference to WandB run {self.run.name}: {err}")


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
