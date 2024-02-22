"""
Callbacks to use with the Engine class and customize the training loop.
"""
from __future__ import annotations

import dataclasses as D
import enum as E
import typing as T
from pprint import pformat

import numpy as np
import typing_extensions as TX
from tqdm.auto import tqdm

import unipercept.log
import unipercept.state

if T.TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from unipercept.engine import EngineParams

__all__ = [
    "State",
    "Signal",
    "Event",
    "Event",
    "CallbackDispatcher",
    "CallbackType",
    "Delegate",
    "FlowCallback",
    "ProgressCallback",
    "Logger",
    "ConditionalStoppingCallback",
    "EarlyStoppingCallback",
]

_logger = unipercept.log.get_logger(__name__)


@D.dataclass(kw_only=True)
class State:
    """
    State class that holds the state of the training loop.
    The state is saved as part of the training checkpoint and can be used to resume
    training from a saved checkpoint.
    """

    # Epochs and steps
    epoch: float = 0.0
    step: int = 0

    # Training stage index
    stage: int = 0

    # Accumulation steps in current session
    gradient_accumulation: int = 1

    # Materialized step values
    train_steps: int = 0
    eval_steps: int | None = None
    save_steps: int | None = None
    logging_steps: int | None = None

    # Flops
    total_flops: float = 0

    # Logs
    log_history: list[dict[str, float]] = D.field(
        default_factory=list, repr=False, compare=False
    )

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


@D.dataclass
class Signal:
    """
    A class that handles the ``Engine`] control flow. This class is used by the [`Callback`` to activate some
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
        """Internal method that resets the signal variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the signal variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the signal variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


######################
# EVENTS DEFINITIONS #
######################


class Event(E.StrEnum):
    """
    Events that are triggered during the training loop.
    """

    ON_CREATE = E.auto()
    ON_TRACKERS_SETUP = E.auto()
    ON_TRAIN_BEGIN = E.auto()
    ON_TRAIN_END = E.auto()
    ON_TRAIN_EPOCH_BEGIN = E.auto()
    ON_TRAIN_EPOCH_END = E.auto()
    ON_TRAIN_STEP_BEGIN = E.auto()
    ON_TRAIN_SUBSTEP_END = E.auto()
    ON_TRAIN_STEP_END = E.auto()
    ON_EVALUATE = E.auto()
    ON_PREDICT = E.auto()
    ON_INFERENCE_END = E.auto()
    ON_INFERENCE_BEGIN = E.auto()
    ON_INFERENCE_STEP = E.auto()
    ON_SAVE = E.auto()
    ON_LOG = E.auto()


class CallbackDispatcher:
    """
    Baseclass for callbacks that dispatches events to functions with the same name, e.g. the event 'on_train_begin' is
    dispatched to ``on_train_begin(self, ...)``.
    """

    __slots__ = ()

    def __call__(
        self,
        event: Event,
        params: EngineParams,
        state: State,
        control: Signal,
        **kwargs,
    ):
        """
        Override this method to implement your own logic. By default, it switches the control flow to the correct event.
        """

        event_name = Event(event).value
        handler: T.Callable[..., Signal | None] = getattr(self, event_name)
        handler(params, state, control, **kwargs)

    def on_create(self, params: EngineParams, state: State, control: Signal):
        """
        Event called at the end of the initialization of the ``Engine``.
        """

    def on_trackers_setup(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        session_id: str,
        **kwargs,
    ):
        """
        Event called just before the initialization of the trackers, such that the user can pass additional keyword
        arguments to the tracker by modifying the ``init_kwargs`` dictionary.
        """

    def on_train_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called at the beginning of training.
        """

    def on_train_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called at the end of training.
        """

    def on_train_epoch_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called at the beginning of an epoch.
        """

    def on_train_epoch_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called at the end of an epoch.
        """

    def on_train_step_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """

    def on_train_substep_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called at the end of an substep during gradient accumulation.
        """

    def on_train_step_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """

    def on_evaluate(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called after an evaluation phase.
        """

    def on_predict(
        self, params: EngineParams, state: State, control: Signal, metrics, **kwargs
    ):
        """
        Event called after a successful prediction.
        """

    def on_save(self, params: EngineParams, state: State, control: Signal, **kwargs):
        """
        Event called after a checkpoint save.
        """

    def on_log(self, params: EngineParams, state: State, control: Signal, **kwargs):
        """
        Event called after logging the last logs.
        """

    def on_inference_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called after a prediction step.
        """

    def on_inference_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called after a prediction step.
        """

    def on_inference_step(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        """
        Event called after a prediction step.
        """


########################################
# DELEGATE CLASS AND CALLBACK PROTOCOL #
########################################


@T.runtime_checkable
class CallbackProtocol(T.Protocol):
    def __call__(
        self,
        event: Event,
        params: EngineParams,
        state: State,
        control: Signal,
        **kwargs,
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

    def __call__(
        self,
        event: Event,
        params: EngineParams,
        state: State,
        control: Signal,
        **kwargs,
    ) -> Signal:
        if self._verbose:
            _logger.debug(f"Event {event} triggered.")

        for cb in self._seq:
            result = cb(
                event,
                params,
                state,
                control,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


####################################
# DEFAULT CALLBACKS IMPLEMENTATION #
####################################


class InternalCallback(CallbackDispatcher):
    @TX.override
    def on_train_step_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False

    @TX.override
    def on_save(self, params: EngineParams, state: State, control: Signal, **kwargs):
        control.should_save = False

    @TX.override
    def on_log(self, params: EngineParams, state: State, control: Signal, **kwargs):
        control.should_log = False

    @TX.override
    def on_evaluate(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        control.should_evaluate = False

    @TX.override
    def on_train_epoch_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
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

    @TX.override
    def on_train_step_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        # Log
        if state.step == 1 and params.logging_first_step:
            control.should_log = True
        if self.should_run(state.step, state.logging_steps):
            control.should_log = True
        if (
            self.should_run(state.step, state.eval_steps)
            and params.eval_delay <= state.step
        ):
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

    @TX.override
    def on_train_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if unipercept.state.check_main_process(True):
            self.training_bar = tqdm(
                desc="Training", total=state.train_steps, dynamic_ncols=True
            )
        self.current_step = 0

    @TX.override
    def on_train_step_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if unipercept.state.check_main_process(True):
            assert (
                self.training_bar is not None
            ), f"Training bar does not exist at step {state.step}."
            self.training_bar.update(state.step - self.current_step)
            self.current_step = state.step

    @TX.override
    def on_inference_step(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        loader: DataLoader,
        **kwargs,
    ):
        if unipercept.state.check_main_process(True) and len(loader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    desc="Inference",
                    total=len(loader),
                    leave=self.training_bar is None,
                    dynamic_ncols=True,
                )
            self.prediction_bar.update(1)

    @TX.override
    def on_inference_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if unipercept.state.check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @TX.override
    def on_evaluate(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if unipercept.state.check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @TX.override
    def on_predict(self, params: EngineParams, state: State, control: Signal, **kwargs):
        if unipercept.state.check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @TX.override
    def on_log(
        self, params: EngineParams, state: State, control: Signal, logs=None, **kwargs
    ):
        if unipercept.state.check_main_process(True) and self.training_bar is not None:
            # self.training_bar.write(str(logs))
            pass

    @TX.override
    def on_train_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if unipercept.state.check_main_process(True):
            if self.training_bar is not None:
                self.training_bar.close()
            self.training_bar = None


class Logger(CallbackDispatcher):
    """
    A bare ``Callback`` that just prints the logs.
    """

    @TX.override
    def on_log(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        logs: dict,
        **kwargs,
    ):
        _ = logs.pop("total_flops", None)
        if unipercept.state.check_main_process(True):
            _logger.info("Logs: %s", pformat(logs, indent=0, compact=True))
            _logger.info(
                "State: %s", pformat(state.state_dict(), indent=0, compact=True)
            )


class ConditionalStoppingCallback(CallbackDispatcher):
    """
    A ``Callback`` that performs stopping based on a parameter condition.
    """

    def __init__(
        self, metric_name: str, maximize: bool, threshold: float, patience: int = 1
    ):
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")

        self.metric_name = metric_name
        self.maximize = maximize
        self.threshold = threshold
        self.patience = patience
        self.patience_counter = 0

    @TX.override
    def on_evaluate(
        self, params: EngineParams, state: State, control: Signal, metrics, **kwargs
    ):
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            _logger.warning(
                f"conditional stopping did not find {self.metric_name} so early stopping"
                " is disabled"
            )
            return

        operator = np.less if self.maximize else np.greater
        if operator(metric_value, self.threshold):
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        if self.patience_counter >= self.patience:
            _logger.info(
                "Conditional stopping triggered for parameter %s", self.metric_name
            )
            control.should_training_stop = True


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
    in ``EngineState``. Note that if the ``config`` argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: T.Optional[float] = 0.0,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        metric_value,
        metric_maximize: bool = True,
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

    @TX.override
    def on_train_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        assert (
            params.metric is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"

    @TX.override
    def on_evaluate(
        self, params: EngineParams, state: State, control: Signal, metrics, **kwargs
    ):
        if not unipercept.state.check_main_process():
            return

        metric_to_check = params.metric
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            _logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(params, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
