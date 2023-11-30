"""
Callbacks to use with the Trainer class and customize the training loop.
"""
from __future__ import annotations

import dataclasses as D
import enum as E
import typing as T
from pprint import pformat
from typing import Any, Mapping

import accelerate
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from transformers.trainer_utils import IntervalStrategy, has_length
from typing_extensions import override

import unipercept.utils.logutils
import unipercept.utils.state
from unipercept.trainer.config import TrainConfig

if T.TYPE_CHECKING:
    import accelerate.data_loader
    import accelerate.optimizer
    import accelerate.scheduler
    import timm.scheduler.scheduler
    import torch.distributed.fsdp as fsdp
    import torch.nn as nn
    import torch.nn.parallel as ll
    from torch.utils.data import DataLoader

    from ._scheduler import SchedulerFactory
    from .config import TrainConfig

__all__ = [
    "TrainState",
    "Signal",
    "Event",
    "EventType",
    "CallbackDispatcher",
    "CallbackType",
    "Delegate",
    "FlowCallback",
    "ProgressCallback",
    "Logger",
]

_logger = unipercept.utils.logutils.get_logger(__name__)


@D.dataclass(kw_only=True)
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
    log_history: list[dict[str, float]] = D.field(default_factory=list, repr=False, compare=False)

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
        keys = [f.name for f in D.fields(self)]

        return {k: v for k, v in D.asdict(self).items() if k in keys}

    def load_state_dict(self, state_dict: dict[str, T.Any]):
        self.__init__(**state_dict)

    def reset(self):
        self.__init__()

    # @property
    # def checkpoint_pattern(self) -> re.Pattern:
    #     if self.trial_name is None:
    #         raise ValueError("Trial name is not set.")
    #     return get_checkpoint_pattern(self.trial_name)

    # @property
    # def checkpoint_prefix(self):
    #     if self.trial_name is None:
    #         raise ValueError("Trial name is not set.")
    #     return get_checkpoint_prefix(self.trial_name)

    # @property
    # def checkpoint_name_step(self) -> str:
    #     if self.trial_name is None:
    #         raise ValueError("Trial name is not set.")
    #     return get_checkpoint_name(self.trial_name, f"{self.step:12d}")

    # @property
    # def checkpoint_name_best(self) -> str:
    #     """
    #     Return the path to the best checkpoint.
    #     """
    #     if self.trial_name is None:
    #         raise ValueError("Trial name is not set.")
    #     return get_checkpoint_name(self.trial_name, "best")


# @D.dataclass(slots=True, frozen=True)
# class Session(T.Generic[_I]):
#     model: nn.Module | fsdp.FullyShardedDataParallel | ll.DistributedDataParallel | ll.DataParallel
#     optimizer: accelerate.scheduler.AcceleratedScheduler
#     scheduler: timm.scheduler.scheduler.Scheduler
#     loader: torch.utils.data.DataLoader[_I]


@D.dataclass
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


class Event(E.StrEnum):
    """
    Events that are triggered during the training loop.
    """

    ON_INIT_END = E.auto()
    ON_TRAIN_BEGIN = E.auto()
    ON_TRAIN_END = E.auto()
    ON_TRAIN_EPOCH_BEGIN = E.auto()
    ON_TRAIN_EPOCH_END = E.auto()
    ON_TRAIN_STEP_BEGIN = E.auto()
    ON_TRAIN_SUBSTEP_END = E.auto()
    ON_TRAIN_STEP_END = E.auto()
    ON_EVALUATE = E.auto()
    ON_PREDICT = E.auto()
    ON_SAVE = E.auto()
    ON_LOG = E.auto()
    ON_INFERENCE_STEP = E.auto()


EventType: T.TypeAlias = (
    Event
    | T.Literal["on_init_end"]
    | T.Literal["on_train_begin"]
    | T.Literal["on_train_end"]
    | T.Literal["on_train_epoch_begin"]
    | T.Literal["on_train_epoch_end"]
    | T.Literal["on_train_step_begin"]
    | T.Literal["on_train_substep_end"]
    | T.Literal["on_train_step_end"]
    | T.Literal["on_evaluate"]
    | T.Literal["on_predict"]
    | T.Literal["on_save"]
    | T.Literal["on_log"]
    | T.Literal["on_inference_step"]
)


@T.runtime_checkable
class CallbackProtocol(T.Protocol):
    def __call__(
        self, event: EventType, config: TrainConfig, state: TrainState, control: Signal, **kwargs
    ) -> Signal | None:
        ...


CallbackType: T.TypeAlias = CallbackProtocol | type[CallbackProtocol]


class Delegate:
    """
    Handler for managing all the callbacks to be executed inside the training loop.
    """

    def __init__(self, callbacks: T.Sequence[CallbackType], *, verbose=False):
        self._verbose = verbose
        self._seq = [InternalCallback()]
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
        if isinstance(__cb, InternalCallback) or __cb == InternalCallback:
            raise ValueError("``InternalCallback`` cannot be removed.")
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
        if isinstance(__cb, InternalCallback) or __cb == InternalCallback:
            raise ValueError("``InternalCallback`` cannot be removed.")
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
        if self._verbose:
            _logger.debug(f"Event {event} triggered.")

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


class CallbackDispatcher:
    """
    Baseclass for callbacks that dispatches events to functions with the same name, e.g. the event 'on_train_begin' is
    dispatched to ``on_train_begin(self, ...)``.
    """

    __slots__ = ()

    def __call__(self, event: EventType, config: TrainConfig, state: TrainState, control: Signal, **kwargs) -> None:
        """
        Override this method to implement your own logic. By default, it switches the control flow to the correct event.
        """

        event_name = Event(event).value
        try:
            handler: T.Callable[..., None] = getattr(self, event_name)
        except AttributeError:
            return None  # no control
        return handler(config, state, control, **kwargs)

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

    def on_train_epoch_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_train_epoch_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_train_step_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_train_substep_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    def on_train_step_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
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

    def on_inference_step(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        """
        Event called after a prediction step.
        """
        pass


class InternalCallback(CallbackDispatcher):
    @override
    def on_train_step_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False

    @override
    def on_save(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        control.should_save = False

    @override
    def on_log(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        control.should_log = False

    @override
    def on_evaluate(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        control.should_evaluate = False

    @override
    def on_train_epoch_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        control.should_epoch_stop = False


class FlowCallback(CallbackDispatcher):
    """
    A ``Callback`` that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    @staticmethod
    def should_run(step: int, target: int | None) -> bool:
        if target is None:
            return False
        return step > 0 and step % target == 0

    @override
    def on_train_step_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        # Log
        if state.step == 1 and config.logging_first_step:
            control.should_log = True
        if self.should_run(state.step, state.logging_steps):
            control.should_log = True
        if self.should_run(state.step, state.eval_steps) and config.eval_delay <= state.step:
            control.should_evaluate = True
        if self.should_run(state.step, state.save_steps):
            control.should_save = True
        if state.step >= state.train_steps:
            control.should_training_stop = True
        return control


class ProgressCallback(CallbackDispatcher):
    """
    A ``Callback`` that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.prediction_bar = None

    @override
    def on_train_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        if unipercept.utils.state.check_main_process(True):
            self.training_bar = tqdm(desc="Training", total=state.train_steps, dynamic_ncols=True)
        self.current_step = 0

    @override
    def on_train_step_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        if unipercept.utils.state.check_main_process(True):
            assert self.training_bar is not None, f"Training bar does not exist at step {state.step}."
            self.training_bar.update(state.step - self.current_step)
            self.current_step = state.step

    @override
    def on_inference_step(
        self, config: TrainConfig, state: TrainState, control: Signal, *, loader: DataLoader, **kwargs
    ):
        if unipercept.utils.state.check_main_process(True) and has_length(loader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    desc="Inference", total=len(loader), leave=self.training_bar is None, dynamic_ncols=True
                )
            self.prediction_bar.update(1)

    @override
    def on_evaluate(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        if unipercept.utils.state.check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @override
    def on_predict(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        if unipercept.utils.state.check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @override
    def on_log(self, config: TrainConfig, state: TrainState, control: Signal, logs=None, **kwargs):
        if unipercept.utils.state.check_main_process(True) and self.training_bar is not None:
            _ = logs.pop("total_flops", None)
            self.training_bar.write(str(logs))

    @override
    def on_train_end(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        if unipercept.utils.state.check_main_process(True):
            self.training_bar.close()
            self.training_bar = None


class Logger(CallbackDispatcher):
    """
    A bare ``Callback`` that just prints the logs.
    """

    @override
    def on_log(self, config: TrainConfig, state: TrainState, control: Signal, *, logs: dict, **kwargs):
        _ = logs.pop("total_flops", None)
        if unipercept.utils.state.check_main_process(True):
            _logger.info("Logs: %s", pformat(logs, indent=0, compact=True))
            _logger.info("State: %s", pformat(state.state_dict(), indent=0, compact=True))


class EarlyStoppingCallback(CallbackDispatcher):
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

    def check_metric_value(
        self, config: TrainConfig, state: TrainState, control: Signal, *, metric_value, metric_maximize: bool = True
    ):
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
    def on_train_begin(self, config: TrainConfig, state: TrainState, control: Signal, **kwargs):
        assert config.metric is not None, "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            config.evaluation_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    @override
    def on_evaluate(self, config: TrainConfig, state: TrainState, control: Signal, metrics, **kwargs):
        metric_to_check = config.metric
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
