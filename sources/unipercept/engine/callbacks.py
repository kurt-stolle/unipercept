"""
Callbacks to use with the Engine class and customize the training loop.
"""

from __future__ import annotations

import dataclasses as D
import enum as E
import typing as T
from pprint import pformat

import numpy as np
import torch
import typing_extensions as TX
from torch import Tensor, nn
from tqdm.auto import tqdm

from unipercept.log import create_table, get_logger
from unipercept.model import ModelBase, ModelInput, ModelOutput
from unipercept.state import check_main_process

if T.TYPE_CHECKING:
    from accelerate.optimizer import AcceleratedOptimizer
    from timm.scheduler import Scheduler as TimmScheduler
    from torch.utils.data import DataLoader

    from unipercept.engine import EngineParams, Interval, OptimizerFactory
    from unipercept.engine.accelerate import Accelerator

_logger = get_logger(__name__)


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
    step_experiment: int = 0

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

    def register_step(
        self,
        *,
        epoch: int,
        step: int,
        steps_skipped: int,
        steps_in_epoch: int,
        n: int = 1,
    ) -> None:
        """Called when a training step has been performed"""
        self.step += n
        self.step_experiment += n
        self.epoch = float(epoch) + float(step + n + steps_skipped) / float(
            steps_in_epoch
        )

    def register_logs(self, logs: dict[str, T.Any], *, max_history: int) -> None:
        """Called when logs are being pushed"""
        if max_history <= 0:
            return
        self.log_history.append(logs)
        if len(self.log_history) > max_history:
            self.log_history.pop(0)

    def register_training(
        self,
        *,
        logging_steps: int | None,
        eval_steps: int | None,
        save_steps: int | None,
        train_steps: int,
        gradient_accumulation: int,
        best_metric: float | None,
        trial_name: str | None,
        trial_config: dict[str, T.Any] | None,
    ):
        """Called when a training loop is started"""

        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.train_steps = train_steps
        self.gradient_accumulation = gradient_accumulation
        self.best_metric = best_metric
        self.trial_name = trial_name
        self.trial_params = trial_config
        self.step = 0
        self.epoch = 0.0
        self.total_flops = 0
        self.log_history = []


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
    ON_ACCELERATOR_SETUP = E.auto()
    ON_MODEL_SETUP = E.auto()
    ON_TRACKERS_SETUP = E.auto()
    ON_TRAIN_BEGIN = E.auto()
    ON_TRAIN_END = E.auto()
    ON_TRAIN_EPOCH_BEGIN = E.auto()
    ON_TRAIN_EPOCH_END = E.auto()
    ON_TRAIN_STEP_BEGIN = E.auto()
    ON_TRAIN_SUBSTEP_END = E.auto()
    ON_TRAIN_STEP_END = E.auto()
    ON_TRAIN_GRADIENTS = E.auto()
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
        Override this method to implement your own logic.
        By default, it switches the control flow to the correct event.
        """

        event_name = Event(event).value
        if not hasattr(self, event_name):
            return
        handler: T.Callable[..., Signal | None] = getattr(self, event_name)
        handler(params, state, control, **kwargs)

    if T.TYPE_CHECKING:
        # We define the following methods only for reference. They can be overriden
        # in derived classes using the given signature.
        def on_create(
            self, params: EngineParams, state: State, control: Signal, **kwargs
        ):
            """
            Event called at the end of the initialization of the ``Engine``.
            """

        def on_model_setup(
            self,
            params: EngineParams,
            state: State,
            control: Signal,
            *,
            model: ModelBase,
            training: bool,
            **kwargs,
        ):
            """
            Event called at the end of the initialization of the model.
            """
            ...

        def on_accelerator_setup(
            self,
            params: EngineParams,
            state: State,
            control: Signal,
            *,
            accelerator: Accelerator,
            **kwargs,
        ):
            """
            Event called at the end of the initialization of the ``Accelerator``.
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
            self,
            params: EngineParams,
            state: State,
            control: Signal,
            *,
            model: ModelBase,
            optimizer: AcceleratedOptimizer,
            scheduler: TimmScheduler,
            **kwargs,
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
            self,
            params: EngineParams,
            state: State,
            control: Signal,
            *,
            model: ModelBase,
            optimizer: AcceleratedOptimizer,
            scheduler: TimmScheduler,
            **kwargs,
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

        def on_train_gradients(
            self,
            params: EngineParams,
            state: State,
            control: Signal,
            *,
            model: nn.Module,
            losses: dict[str, Tensor],
            **kwargs,
        ):
            """
            Event called during training before stepping the optimizer when the gradients
            are available and have been unscaled by the gradient scaler.
            """

        def on_evaluate(
            self, params: EngineParams, state: State, control: Signal, **kwargs
        ):
            """
            Event called after an evaluation phase.
            """

        def on_predict(
            self,
            params: EngineParams,
            state: State,
            control: Signal,
            *,
            metrics,
            **kwargs,
        ):
            """
            Event called after a successful prediction.
            """

        def on_save(
            self, params: EngineParams, state: State, control: Signal, **kwargs
        ):
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


class StatefulCallbackDispatcher(CallbackDispatcher):
    """
    Callback dispatcher that can save and load its state.
    """

    __slots__ = ("__dict__",)  # non-state attributes must be defined as a slot

    def state_dict(self) -> dict[str, T.Any]:
        return self.__dict__

    def load_state_dict(self, state_dict: dict[str, T.Any]):
        self.__dict__ = state_dict

    @TX.override
    def on_accelerator_setup(
        self,
        *args,
        accelerator: Accelerator,
        **kwargs,
    ):
        accelerator.register_for_checkpointing(self)


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
        if check_main_process(True):
            self.training_bar = tqdm(
                desc="Training", total=state.train_steps, dynamic_ncols=True
            )
        self.current_step = 0

    @TX.override
    def on_train_step_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if check_main_process(True):
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
        if check_main_process(True) and len(loader):
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
        if check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @TX.override
    def on_evaluate(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @TX.override
    def on_predict(self, params: EngineParams, state: State, control: Signal, **kwargs):
        if check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @TX.override
    def on_log(
        self, params: EngineParams, state: State, control: Signal, logs=None, **kwargs
    ):
        if check_main_process(True) and self.training_bar is not None:
            # self.training_bar.write(str(logs))
            pass

    @TX.override
    def on_train_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if check_main_process(True):
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
        if check_main_process(True):
            _logger.info("Logs: %s", pformat(logs, indent=0, compact=True))
            _logger.info(
                "State: %s", pformat(state.state_dict(), indent=0, compact=True)
            )


######################
# Stopping callbacks #
######################


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
        if not check_main_process():
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


######################
# Training callbacks #
######################


class GradientClippingCallback(CallbackDispatcher):
    """
    A ``Callback`` that handles gradient clipping during training.
    """

    def __init__(
        self,
        max_norm: float | None = None,
        max_value: float | None = None,
        norm_tracker: nn.Module | None = None,
    ):
        """
        Parameters
        ----------
        max_norm:
            The maximum total norm of all gradients.
        max_value:
            The maximum absolute value of any individual gradient.
        norm_tracker:
            The tracker module to use for tracking the total norm of the gradients.
            If None, the gradient is clipped based only by the ``max_norm`` value.
        """
        self.max_norm = max_norm
        self.max_value = max_value
        self.norm_tracker = norm_tracker
        self.total_norm: Tensor | None = None
        self.step_counter: Tensor | None = None

        assert (
            self.max_norm is None or self.max_norm >= 0
        ), "max_norm must be non-negative or disabled"
        assert (
            self.max_value is None or self.max_value >= 0
        ), "max_value must be non-negative or disabled"
        assert (self.norm_tracker is None) or (
            self.norm_tracker is not None and self.max_norm is not None
        ), "max_norm must be defined when using a tracker"

    @TX.override
    def on_accelerator_setup(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        accelerator: Accelerator,
        **kwargs,
    ):
        if self.norm_tracker is not None:
            accelerator.register_for_checkpointing(self.norm_tracker)
        if self.norm_tracker is not None or self.max_norm is not None:
            self.total_norm = torch.tensor(
                0.0, device=accelerator.device, dtype=torch.float32, requires_grad=False
            )
        self.step_counter = torch.tensor(
            0, device=accelerator.device, dtype=torch.int32, requires_grad=False
        )

    @TX.override
    def on_train_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        assert self.step_counter is not None
        self.step_counter.zero_()

        if self.total_norm is not None:
            self.total_norm.zero_()

        if self.norm_tracker is not None:
            self.norm_tracker.reset()

    @TX.override
    def on_log(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        logs: dict[str, T.Any],
        **kwargs,
    ):
        assert self.step_counter is not None
        if self.step_counter == 0:
            return

        if self.total_norm is not None:
            logs["optimizer/total_norm"] = (self.total_norm / self.step_counter).item()
            self.total_norm.zero_()

        if self.norm_tracker is not None:
            smooth_norm = self.norm_tracker.observe().item()
            logs["optimizer/smooth_norm"] = smooth_norm

        self.step_counter.zero_()

    @TX.override
    def on_train_substep_end(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        step_last_logged: int,
        **kwargs,
    ):
        if self.total_norm is not None:
            self.total_norm += self.total_norm / (1 + state.step - step_last_logged)

    @TX.override
    def on_train_gradients(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        model: nn.Module,
        **kwargs,
    ):
        assert self.step_counter is not None

        model_params = list(model.parameters())
        for p in model_params:
            if p is None or p.grad is None:
                continue
            torch.nan_to_num(p.grad, nan=0.0, posinf=1e8, neginf=-1e8, out=p.grad)

        self.step_counter += 1

        # Clip gradients by value
        if self.max_value is not None:
            nn.utils.clip_grad_value_(model_params, self.max_value)

        # Clip gradients by norm
        if self.max_norm is not None:
            assert self.total_norm is not None

            max_norm: float
            # Read the max norm value (from smoother or directly from params)
            if self.norm_tracker is not None:
                max_norm_obs = self.norm_tracker.observe()
                if not torch.isfinite(max_norm_obs):
                    max_norm = self.max_norm
                else:
                    max_norm = max_norm_obs.item()
                    max_norm = min(max_norm, self.max_norm)
            else:
                max_norm = self.max_norm

            # Apply gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(
                model_params, max_norm, norm_type=2
            )
            total_norm = total_norm.detach()

            # Smooth the gradient norm
            if self.norm_tracker is not None and torch.isfinite(total_norm):
                self.norm_tracker(total_norm)

            # Update the total norm
            self.total_norm.add_(total_norm)


#####################################
# Callbacks for Multi-Task Learning #
#####################################


class UncertaintyLossWeightingCallback(StatefulCallbackDispatcher):
    """
    Implements the Uncertrainty loss weighting algorithm from [1]

    References
    ----------
    [1] Kendall et al., "Multi-Task Learning Using Uncertrainty to Weigh Losses for Scene Geometry and Semantics". CVPR 2018. https://arxiv.org/pdf/1705.07115
    """

    def __init__(self):
        self.loss_weights: Tensor | None = None

    # TODO


class TaskRebalanceCallback(StatefulCallbackDispatcher):
    """
    Implements a task rebalancing callback without optimization.
    """

    __slots__ = ("gamma", "window", "hook_handle", "verbose", "task_names")

    def __init__(
        self,
        tasks: T.Iterable[str | T.Iterable[str]]
        | T.Mapping[str, T.Iterable[str] | str],
        gamma: float = 0.5,
        window: int = 2,
        verbose: bool = False,
        allow_missing: bool = True,
    ):
        self.groups = []

        if isinstance(tasks, T.Mapping):
            task_names = list(tasks.keys())
            tasks = tasks.values()
        else:
            task_names = None

        for task in tasks:
            if isinstance(task, str):
                self.groups.append([task])
            else:
                self.groups.append(list(task))

        if task_names is None:
            task_names = [t[0] for t in self.groups]

        self.task_names = task_names
        self.window = window
        self.gamma = gamma
        self.weights: Tensor | None = None
        self.losses: Tensor | None = None
        self.hook_handle: torch.utils.hooks.RemovableHandle | None = None
        self.verbose = verbose
        self.allow_missing = allow_missing

    @property
    def task_weights(self) -> dict[str, float]:
        return dict(zip(self.task_names, self.weights.tolist()))

    @TX.override
    def on_accelerator_setup(self, *args, accelerator: Accelerator, **kwargs):
        self.weights = torch.full(
            (len(self.groups),),
            torch.nan,
            device=accelerator.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.losses = torch.full(
            (len(self.groups), self.window),
            torch.nan,
            device=accelerator.device,
            dtype=torch.float32,
            requires_grad=False,
        )

    @TX.override
    def on_model_setup(self, *args, model: nn.Module, training: bool, **kwargs):
        if not training:
            return
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

        def apply_weights_hook(
            module: nn.Module,
            inputs: ModelInput,
            outputs: ModelOutput | dict[str, Tensor],
        ) -> ModelOutput | None:
            r"""
            Hook that applies the loss weights in the forward of the model.
            """
            if not module.training:
                return None
            assert self.weights is not None

            if isinstance(outputs, ModelOutput):
                target = outputs.losses
            elif isinstance(outputs, dict):
                target = outputs
            else:
                msg = f"Outputs must be a ModelOutput or a dict, got {type(outputs)}"
                raise ValueError(msg)

            if self.verbose:
                _logger.debug(
                    "%s weights: %s",
                    self.__class__.__name__,
                    self.task_weights,
                )

            for i, group in enumerate(self.groups):
                w = self.weights[i]
                for task in group:
                    if task in target:
                        target[task] = target[task] * w.detach()
                    elif self.allow_missing:
                        pass
                    else:
                        msg = f"Task {task} not found in outputs (keys: {list(target.keys())})"
                        raise ValueError(msg)
            return outputs

        self.hook_handle = model.register_forward_hook(apply_weights_hook)

    @TX.override
    def on_train_begin(self, params, state, control, *, model: nn.Module, **kwargs):
        if self.weights is None or self.losses is None:
            msg = f"{self.__class__.__name__} requires the accelerator to be set up before training."
            raise RuntimeError(msg)

        self.weights.fill_(1.0)
        self.losses.fill_(torch.nan)

    @TX.override
    @torch.no_grad()
    def on_train_step_begin(self, *args, **kwargs):
        r"""
        Compute the weights given the current losses.
        """
        assert self.losses is not None
        if torch.isnan(self.losses).any():
            return
        loss_1, loss_2 = self.losses.chunk(2, dim=-1)
        loss_1 = loss_1.mean(dim=-1)
        loss_2 = loss_2.mean(dim=-1)
        w = (loss_1 / loss_2) * self.gamma

        self.weights = w.softmax(dim=0) * len(self.groups)

    @TX.override
    @torch.no_grad()
    def on_train_step_end(
        self, params, state, control, *, losses: dict[str, Tensor], **kwargs
    ):
        assert self.losses is not None
        self.losses = self.losses.roll(1, dims=-1)
        for i, group in enumerate(self.groups):
            group_losses = []
            for task in group:
                if task in losses:
                    group_losses.append(losses[task].detach())
                elif self.allow_missing:
                    pass
                else:
                    msg = (
                        f"Task {task} not found in losses (keys: {list(losses.keys())})"
                    )
                    raise ValueError(msg)
            self.losses[i, 0] = torch.stack(group_losses).sum()


class TaskParameterRebalanceCallback(StatefulCallbackDispatcher):
    """
    Implements a task parameter rebalancing callback.
    """

    def __init__(self, optimizer: OptimizerFactory):
        self.optimizer_factory = optimizer
        self.optimizer: Optimizer | None = None
        self.model: ModelBase | None = None

    def virtual_step(self, train_x, train_y, alpha, model_optim):
        """
        Compute unrolled network theta' (virtual step)
        """

        # forward & compute loss
        if type(train_x) == list:  # multi-domain setting [many-to-many]
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:  # single-domain setting [one-to-many]
            train_pred = self.model(train_x)

        train_loss = self.model_fit(train_pred, train_y)

        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])

        # compute gradient
        gradients = torch.autograd.grad(loss, self.model.parameters())

        # do virtual step (update gradient): theta' = theta - alpha * sum_i lambda_i * L_i(f_theta(x_i), y_i)
        with torch.no_grad():
            for weight, weight_, grad in zip(
                self.model.parameters(), self.model_.parameters(), gradients
            ):
                if (
                    "momentum" in model_optim.param_groups[0].keys()
                ):  # used in SGD with momentum
                    m = (
                        model_optim.state[weight].get("momentum_buffer", 0.0)
                        * model_optim.param_groups[0]["momentum"]
                    )
                else:
                    m = 0
                weight_.copy_(
                    weight
                    - alpha
                    * (m + grad + model_optim.param_groups[0]["weight_decay"] * weight)
                )

    def unrolled_backward(self, train_x, train_y, val_x, val_y, alpha, model_optim):
        """
        Compute un-rolled loss and backward its gradients
        """

        # do virtual step (calc theta`)
        self.virtual_step(train_x, train_y, alpha, model_optim)

        # define weighting for primary tasks (with binary weights)
        pri_weights = []
        for t in self.train_tasks:
            if t in self.pri_tasks:
                pri_weights += [1.0]
            else:
                pri_weights += [0.0]

        # compute validation data loss on primary tasks
        if type(val_x) == list:
            val_pred = [self.model_(x, t) for t, x in enumerate(val_x)]
        else:
            val_pred = self.model_(val_x)
        val_loss = self.model_fit(val_pred, val_y)
        loss = sum([w * val_loss[i] for i, w in enumerate(pri_weights)])

        # compute hessian via finite difference approximation
        model_weights_ = tuple(self.model_.parameters())
        d_model = torch.autograd.grad(loss, model_weights_, allow_unused=True)
        hessian = self.compute_hessian(d_model, train_x, train_y)

        # update final gradient = - alpha * hessian
        with torch.no_grad():
            for mw, h in zip([self.meta_weights], hessian):
                mw.grad = -alpha * h

    def compute_hessian(self, d_model, train_x, train_y):
        norm = torch.cat([w.view(-1) for w in d_model]).norm()
        eps = 0.01 / norm

        # \theta+ = \theta + eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_p = torch.autograd.grad(loss, self.meta_weights)

        # \theta- = \theta - eps * d_model
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p -= 2 * eps * d

        if type(train_x) == list:
            train_pred = [self.model(x, t) for t, x in enumerate(train_x)]
        else:
            train_pred = self.model(train_x)
        train_loss = self.model_fit(train_pred, train_y)
        loss = sum([w * train_loss[i] for i, w in enumerate(self.meta_weights)])
        d_weight_n = torch.autograd.grad(loss, self.meta_weights)

        # recover theta
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), d_model):
                p += eps * d

        hessian = [(p - n) / (2.0 * eps) for p, n in zip(d_weight_p, d_weight_n)]
        return hessian


###############################
# Precise batch norm callback #
###############################


class PreciseBatchNormCallback(CallbackDispatcher):
    """
    Runs the precise batch norm algorithm to convergence.

    See Also
    --------
    - https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/precise_bn.py
    """

    __slots__ = ("interval", "iterations", "show_progress")

    def __init__(self, interval: Interval, iterations: int, show_progress=True):
        """
        Parameters
        ----------
        interval:
            The interval at which to run the precise batch norm algorithm.
        iterations:
            The number of iterations to run the precise batch norm algorithm.
        show_progress:
            Whether to show the progress bar.
        """

        self.interval = interval
        self.iterations = iterations
        self.show_progress = show_progress

    def compute_precise_batchnorm(self, model: nn.Module, loader: DataLoader):
        from fvcore.nn.precise_bn import update_bn_stats

        _logger.info("Computing precise batch norm statistics...")
        update_bn_stats(model, loader, self.iterations, progress=self.show_progress)

    @TX.override
    def on_train_step_end(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        losses: dict[str, Tensor],
        model: ModelBase,
        loader: DataLoader,
        **kwargs,
    ):
        if self.interval.unit != "steps":
            return
        if state.step % self.interval.amount > 0:
            return

        self.compute_precise_batchnorm(model, loader)

    @TX.override
    def on_train_epoch_end(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        model: ModelBase,
        loader: DataLoader,
        **kwargs,
    ):
        if self.interval.unit != "epochs":
            return

        if round(state.epoch) % self.interval.amount > 0:
            return
        self.compute_precise_batchnorm(model, loader)
