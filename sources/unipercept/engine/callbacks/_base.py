"""
Callbacks to use with the Engine class and customize the training loop.
"""

from __future__ import annotations

import dataclasses as D
import enum as E
import typing as T

import typing_extensions as TX
from torch import Tensor, nn

from unipercept.log import logger
from unipercept.model import ModelBase

if T.TYPE_CHECKING:
    from accelerate.optimizer import AcceleratedOptimizer
    from timm.scheduler import Scheduler as TimmScheduler

    from unipercept.engine import (
        EngineParams,
        ParameterDefinition,
    )
    from unipercept.engine.accelerate import Accelerator

__all__ = [
    "CallbackDispatcher",
    "CallbackProtocol",
    "CallbackType",
    "Delegate",
    "Event",
    "Signal",
    "State",
]


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
    trial_params: dict[str, str | float | int | bool] | None = None

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
    ON_OPTIMIZER_SETUP = E.auto()
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

        def on_optimizer_setup(
            self,
            params: EngineParams,
            state: State,
            control: Signal,
            *,
            stage: EngineStage,
            extra_params: list[ParameterDefinition],
            **kwargs,
        ):
            pass

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
    ) -> Signal | None: ...


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

        if not any("FlowCallback" in type(cb).__name__ for cb in self._seq):
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
            raise ValueError(f"Callback of type {__cb} not found.")
        for cb in self._seq:
            if cb == __cb:
                self._seq.remove(cb)
                return cb
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
            logger.debug(f"Event {event} triggered.")

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


class InternalCallback(CallbackDispatcher):
    @T.override
    def on_train_step_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False

    @T.override
    def on_save(self, params: EngineParams, state: State, control: Signal, **kwargs):
        control.should_save = False

    @T.override
    def on_log(self, params: EngineParams, state: State, control: Signal, **kwargs):
        control.should_log = False

    @T.override
    def on_evaluate(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        control.should_evaluate = False

    @T.override
    def on_train_epoch_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        control.should_epoch_stop = False
