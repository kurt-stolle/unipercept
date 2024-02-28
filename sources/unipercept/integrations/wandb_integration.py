"""
Weights & Biases integration.

See: https://wandb.ai
"""

from __future__ import annotations

import dataclasses as D
import enum as E
import functools
import typing as T

import torch.nn as nn
import typing_extensions as TX
import wandb
import wandb.errors

from unipercept import file_io
from unipercept.config import get_env
from unipercept.engine import EngineParams
from unipercept.engine.callbacks import CallbackDispatcher, Signal, State
from unipercept.log import get_logger
from unipercept.state import on_main_process
from unipercept.utils.time import ProfileAccumulator

if T.TYPE_CHECKING:
    from wandb.sdk.wandb_run import Run as WandBRun


_logger = get_logger(__name__)

WANDB_RUN_PREFIX = "wandb-run://"  # prefix for WandB run URIs


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


def delete_run(*, id: str) -> None:
    """
    Delete a run (if it exists)
    """

    try:
        run = wandb.Api().run(id)
    except wandb.errors.CommError as e:
        _logger.debug("Skipping delete: %s", e)
        return

    _logger.debug("Deleting run: %s", run.id)
    run.delete(delete_artifacts=True)


def read_run(path: str) -> WandBRun:
    """
    Reads a WandB run from a file.
    """

    assert path.startswith(WANDB_RUN_PREFIX)

    _logger.info("Reading W&B configuration from %s", path)

    run_name = path[len(WANDB_RUN_PREFIX) :]
    wandb_api = wandb.Api()
    run = wandb_api.run(run_name)

    return run


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


############################
# WANDB ENGINE INTEGRATION #
############################


def sanitize(s: str) -> str:
    """
    Replaces any characters that are not allowed in a WandB run/artifact/group/uri.
    """
    for c in R"/\#?%:":
        s = s.replace(c, "-")
    return s


##################
# WANDB CALLBACK #
##################


@D.dataclass(slots=True)
class WandBCallback(CallbackDispatcher):
    """
    Extended integration with Weights & Biases.
    """

    watch_model: WandBWatchMode | None | str = D.field(
        default_factory=lambda: get_env(
            str, "UNIPERCEPT_WANDB_WATCH_ENABLED", default=WandBWatchMode.ALL.value
        )
    )
    watch_steps: int | None = D.field(
        default_factory=lambda: get_env(int, "UNIPERCEPT_WANDB_WATCH_INTERVAL"),
        metadata={
            "help": (
                "Interval passed to W&B model watcher. "
                "When set to `None`, the 'logging_steps' attribute is taken from the engine parameters."
            )
        },
    )
    upload_config: bool = D.field(
        default_factory=lambda: get_env(
            bool, "UNIPERCEPT_WANDB_UPLOAD_CONFIG", default=True
        )
    )
    upload_code: bool = D.field(
        default_factory=lambda: get_env(
            bool, "UNIPERCEPT_WANDB_UPLOAD_CODE", default=True
        )
    )
    model_history: int = 1
    state_history: int = 1
    inference_history: int = 0
    tabulate_inference_timings: bool = False

    _session_id: str | None = D.field(
        default=None, init=False
    )  # set when tracking begins

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("Engine not in session")
        return self._session_id

    @property
    def run(self) -> WandBRun:
        """
        Current WandB run.
        """
        assert self.session_id is not None

        run = wandb.run
        if run is None:
            raise RuntimeError("WandB run not initialized")
        return run

    @TX.override
    @skip_no_run
    @on_main_process()
    def on_trackers_setup(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        session_id: str,
        config_path: str,
        **kwargs,
    ):
        self._session_id = session_id

        _logger.info("Tracking current experiment to WandB run %s", self.run.name)

        if self.upload_code:
            self.run.log_code(
                "./sources", include_fn=lambda path, root: path.endswith(".py")
            )
        if self.upload_config:
            assert file_io.isfile(
                config_path
            ), f"Config file {config_path} does not exist!"
            self.run.log_artifact(
                config_path,
                type=ArtifactType.CONFIG.value,
                name=f"{self.run.id}-config",
            )

    @TX.override
    @skip_no_run
    @on_main_process()
    def on_save(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        model_path: str,
        state_path: str,
        **kwargs,
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
    def on_train_begin(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        model: nn.Module,
        **kwargs,
    ):
        if self.watch_model is not None:
            self.run.watch(
                model,
                log=WandBWatchMode(self.watch_model).value,
                log_freq=self.watch_steps
                if self.watch_steps is not None
                else params.logging_steps,
                log_graph=False,
            )

    def _log_model(self, model_path: str):
        try:
            _logger.info(f"Logging model to WandB run {self.run.name}")
            name = f"{self.run.id}-model"
            self.run.log_model(model_path, name=f"{self.run.id}-model")

            artifact = wandb.Api().artifact(
                f"{self.run.entity}/{self.run.project_name()}/{name}:latest",
                type=ArtifactType.MODEL.value,
            )
            artifact_historic_delete(artifact, self.model_history)
        except Exception as err:
            _logger.warning(f"Failed to log model to WandB run {self.run.name}: {err}")

    def _log_state(self, state_path: str):
        try:
            _logger.info(f"Logging engine state to WandB: {self.run.name}")
            state_artifact = wandb.Artifact(
                name=f"{self.run.id}-state",
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
            _logger.warning(
                f"Failed to log state to WandB self.run {self.run.name}: {err}"
            )

    def _log_profiling(self, key: str, timings: ProfileAccumulator):
        run = wandb.run
        assert run is not None, "WandB run not initialized"
        run.log({key: wandb.Table(dataframe=timings.to_summary())}, commit=False)

    def _log_inference(self, results_path: str):
        try:
            _logger.info("Logging evaluation outcomes to WandB")
            artifact = self.run.log_artifact(
                results_path,
                name=f"{self.run.name}-inference",
                type=ArtifactType.INFERENCE.value,
            )
            artifact.wait()

            artifact_historic_delete(artifact, self.inference_history)
        except Exception as err:
            _logger.warning(
                f"Failed to log inference to WandB run {self.run.name}: {err}"
            )


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
