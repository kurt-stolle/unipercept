"""
Callbacks to use with the Trainer class and customize the training loop.
"""
from __future__ import annotations

import dataclasses
import enum
import functools
import json
import re
import types
import typing as T
from dataclasses import dataclass, field
from typing import Any, Mapping

import accelerate
import numpy as np
import torch.nn as nn
import torch.optim as optim
import uniutils.logutils
from tqdm.auto import tqdm
from transformers.trainer_utils import IntervalStrategy, has_length
from typing_extensions import override
from uniutils.state import check_main_process

if T.TYPE_CHECKING:
    import accelerate.data_loader
    import accelerate.optimizer
    import accelerate.scheduler
    import timm.scheduler.scheduler
    import torch.distributed.fsdp as fsdp
    import torch.nn as nn
    import torch.nn.parallel as ll
    import torch.optim as optim
    import torch.utils.data

    from ._scheduler import SchedulerFactory
    from .config import TrainConfig

__all__ = [
    "TrainState",
    "Signal",
    "Event",
    "EventType",
    "Callback",
    "CallbackType",
    "Delegate",
    "FlowCallback",
    "ProgressCallback",
    "Logger",
]

_logger = uniutils.logutils.get_logger(__name__)
_I = T.TypeVar("_I")

_CHECKPOINT_PREFIX = "checkpoint"


def get_checkpoint_prefix(trial_name: str) -> str:
    return f"{_CHECKPOINT_PREFIX}-{trial_name}"


@functools.lru_cache(maxsize=None)
def get_checkpoint_pattern(trial_name: str) -> re.Pattern:
    return re.compile(r"^" + get_checkpoint_prefix(trial_name) + r"\-(\d+)$")


def get_checkpoint_name(trial_name: str, name: str) -> str:
    return f"{get_checkpoint_prefix(trial_name)}-{name}"


@dataclass(kw_only=True)
class TrainState:
    # Epochs and steps
    epoch: float = 0.0
    step: int = 0

    # Materialized step values
    train_steps: int = 0
    eval_steps: int | None = None
    save_steps: int | None = None
    logging_steps: int | None = None

    # Flops
    total_flops: float = 0

    # Logs
    log_history: list[dict[str, float]] | None = None

    # Metrics
    best_metric: float | None = None

    # HP search
    is_search: bool = False
    trial_name: str | None = None
    trial_params: dict[str, T.Union[str, float, int, bool]] | None = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def state_dict(self):
        keys = [f.name for f in dataclasses.fields(self)]

        return {k: v for k, v in dataclasses.asdict(self).items() if k in keys}

    def load_state_dict(self, state_dict: dict[str, T.Any]):
        self.__init__(**state_dict)

    def reset(self):
        self.__init__()

    @property
    def checkpoint_pattern(self) -> re.Pattern:
        if self.trial_name is None:
            raise ValueError("Trial name is not set.")
        return get_checkpoint_pattern(self.trial_name)

    @property
    def checkpoint_prefix(self):
        if self.trial_name is None:
            raise ValueError("Trial name is not set.")
        return get_checkpoint_prefix(self.trial_name)

    @property
    def checkpoint_name_step(self) -> str:
        if self.trial_name is None:
            raise ValueError("Trial name is not set.")
        return get_checkpoint_name(self.trial_name, f"{self.step:12d}")

    @property
    def checkpoint_name_best(self) -> str:
        """
        Return the path to the best checkpoint.
        """
        if self.trial_name is None:
            raise ValueError("Trial name is not set.")
        return get_checkpoint_name(self.trial_name, "best")


# @dataclasses.dataclass(slots=True, frozen=True)
# class Session(T.Generic[_I]):
#     model: nn.Module | fsdp.FullyShardedDataParallel | ll.DistributedDataParallel | ll.DataParallel
#     optimizer: accelerate.scheduler.AcceleratedScheduler
#     scheduler: timm.scheduler.scheduler.Scheduler
#     loader: torch.utils.data.DataLoader[_I]


@dataclass
class Signal:
    """
    A class that handles the ``Trainer`] control flow. This class is used by the [`Callback`` to activate some
    switches in the training loop.

    Parameters
        ----------
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
    """

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class Event(enum.StrEnum):
    """
    Events that are triggered during the training loop.
    """

    ON_INIT_END = enum.auto()
    ON_TRAIN_BEGIN = enum.auto()
    ON_TRAIN_END = enum.auto()
    ON_EPOCH_BEGIN = enum.auto()
    ON_EPOCH_END = enum.auto()
    ON_STEP_BEGIN = enum.auto()
    ON_SUBSTEP_END = enum.auto()
    ON_STEP_END = enum.auto()
    ON_EVALUATE = enum.auto()
    ON_PREDICT = enum.auto()
    ON_SAVE = enum.auto()
    ON_LOG = enum.auto()
    ON_PREDICTION_STEP = enum.auto()


EventType: T.TypeAlias = (
    Event
    | T.Literal["on_init_end"]
    | T.Literal["on_train_begin"]
    | T.Literal["on_train_end"]
    | T.Literal["on_epoch_begin"]
    | T.Literal["on_epoch_end"]
    | T.Literal["on_step_begin"]
    | T.Literal["on_substep_end"]
    | T.Literal["on_step_end"]
    | T.Literal["on_evaluate"]
    | T.Literal["on_predict"]
    | T.Literal["on_save"]
    | T.Literal["on_log"]
    | T.Literal["on_prediction_step"]
)


class _CallbackProtocol(T.Protocol):
    def __call__(
        self, event: EventType, config: TrainConfig, state: TrainState, control: Signal, **kwargs
    ) -> Signal | None:
        ...


CallbackType: T.TypeAlias = _CallbackProtocol | type[_CallbackProtocol]


class Delegate:
    """
    Handler for managing all the callbacks to be executed inside the training loop.
    """

    def __init__(self, callbacks: T.Sequence[CallbackType]):
        self._seq = []
        self._session = None
        for cb in callbacks:
            self.add(cb)

        if not any(isinstance(cb, FlowCallback) for cb in self._seq):
            raise ValueError("``FlowCallback`` is required in the callbacks.")

    def add(self, __cb: CallbackType, /):
        if isinstance(__cb, type):
            self._seq.append(__cb())
        else:
            self._seq.append(__cb)

    def pop(self, __cb: CallbackType, /):
        if isinstance(__cb, type):
            for cb in self._seq:
                if isinstance(cb, __cb):
                    self._seq.remove(cb)
                    return cb
            else:
                raise ValueError(f"Callback of type {__cb} not found.")
        else:
            for cb in self._seq:
                if cb == __cb:
                    self._seq.remove(cb)
                    return cb
            else:
                raise ValueError(f"Callback {__cb} not found.")

    def remove(self, __cb: CallbackType, /):
        if isinstance(__cb, type):
            for cb in self._seq:
                if isinstance(cb, __cb):
                    self._seq.remove(cb)
        else:
            self._seq.remove(__cb)

    @property
    def list(self):
        return "\n".join(cb.__class__.__name__ for cb in self._seq)

    def __call__(self, event: EventType, config: TrainConfig, state: TrainState, control: Signal, **kwargs) -> Signal:
        for cb in self._seq:
            result = cb(
                event,
                config,
                state,
                control,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class CallbackMeta(type):
    """
    Metaclass for callbacks.
    """

    pass


class Callback(metaclass=CallbackMeta):
    """
    Baseclass for callbacks. All callbacks subclass this class and override the methods they need, or just directly override __call__.
    """

    __slots__ = ()

    def __call__(self, event: EventType, config: TrainConfig, state: TrainState, control: Signal, **kwargs) -> None:
        """
        Override this method to implement your own logic. By default, it switches the control flow to the correct event.
        """

        match Event(event):
            case Event.ON_INIT_END:
                return self.on_init_end(config, state, control, **kwargs)
            case Event.ON_TRAIN_BEGIN:
                return self.on_train_begin(config, state, control, **kwargs)
            case Event.ON_TRAIN_END:
                return self.on_train_end(config, state, control, **kwargs)
            case Event.ON_EPOCH_BEGIN:
                return self.on_epoch_begin(config, state, control, **kwargs)
            case Event.ON_EPOCH_END:
                return self.on_epoch_end(config, state, control, **kwargs)
            case Event.ON_STEP_BEGIN:
                return self.on_step_begin(config, state, control, **kwargs)
            case Event.ON_SUBSTEP_END:
                return self.on_substep_end(config, state, control, **kwargs)
            case Event.ON_STEP_END:
                return self.on_step_end(config, state, control, **kwargs)
            case Event.ON_EVALUATE:
                return self.on_evaluate(config, state, control, **kwargs)
            case Event.ON_PREDICT:
                return self.on_predict(config, state, control, **kwargs)
            case Event.ON_PREDICTION_STEP:
                return self.on_prediction_step(config, state, control, **kwargs)
            case Event.ON_LOG:
                return self.on_log(config, state, control, **kwargs)
            case Event.ON_SAVE:
                return self.on_save(config, state, control, **kwargs)
            case _:
                raise ValueError(f"Event {event} not recognized.")

    def on_init_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the end of the initialization of the ``Trainer``.
        """
        pass

    def on_train_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_substep_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    def on_step_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_predict(self, config: TrainConfig, state: TrainState, control: Signal, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        pass

    def on_save(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    def on_prediction_step(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called after a prediction step.
        """
        pass


class FlowCallback(Callback):
    """
    A ``Callback`` that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    @override
    def on_step_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        # Log
        if state.step == 1 and config.logging_first_step:
            control.should_log = True
        if state.step % state.logging_steps == 0:
            control.should_log = True

        # Evaluate
        if state.step % state.eval_steps == 0 and config.eval_delay <= state.step:
            control.should_evaluate = True

        # Save
        if state.save_steps > 0 and state.step % state.save_steps == 0:
            control.should_save = True

        # End training
        if state.step >= state.train_steps:
            control.should_training_stop = True

        return control


class ProgressCallback(Callback):
    """
    A ``Callback`` that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    @override
    def on_train_begin(self, config, state, control, **kwargs):
        if check_main_process(local=True):
            self.training_bar = tqdm(total=state.train_steps, dynamic_ncols=True)
        self.current_step = 0

    @override
    def on_step_end(self, config, state, control, **kwargs):
        if check_main_process(local=True):
            self.training_bar.update(state.step - self.current_step)
            self.current_step = state.step

    @override
    def on_prediction_step(self, config, state, control, loader=None, **kwargs):
        if state.is_local_process_zero and has_length(loader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(total=len(loader), leave=self.training_bar is None, dynamic_ncols=True)
            self.prediction_bar.update(1)

    @override
    def on_evaluate(self, config, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @override
    def on_predict(self, config, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @override
    def on_log(self, config, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flops", None)
            self.training_bar.write(str(logs))

    @override
    def on_train_end(self, config, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None


class Logger(Callback):
    """
    A bare ``Callback`` that just prints the logs.
    """

    @override
    def on_log(self, config, state, control, *, logs: dict, **kwargs):
        _ = logs.pop("total_flops", None)
        if state.is_local_process_zero:
            print(logs)


class EarlyStoppingCallback(Callback):
    """
    A ``Callback`` that handles early stopping.

    Parameters
        ----------
       early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
       early_stopping_threshold(`float`, *optional*):
            Use with config `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on ``config`` argument *load_best_model_at_end* functionality to set best_metric
    in ``TrainerState``. Note that if the ``config`` argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: T.Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, config, state, control, *, metric_value, metric_maximize: bool = True):
        # best_metric is set by code for load_best_model
        operator = np.greater if metric_maximize else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    @override
    def on_train_begin(self, config, state, control, **kwargs):
        assert config.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert config.metric is not None, "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            config.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    @override
    def on_evaluate(self, config, state, control, metrics, **kwargs):
        metric_to_check = config.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            _logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(config, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
