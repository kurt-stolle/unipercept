import typing as T
from pprint import pformat

from torch.utils.data import DataLoader
from tqdm import tqdm

from unipercept.log import logger
from unipercept.state import check_main_process

from .._params import EngineParams
from ._base import CallbackDispatcher, Signal, State

__all__ = ["FlowCallback", "ProgressCallback", "Logger"]


class FlowCallback(CallbackDispatcher):
    """
    A ``Callback`` that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    @staticmethod
    def should_run(step: int, target: int | None) -> bool:
        if target is None:
            return False
        return step > 0 and step % target == 0

    @T.override
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

    @T.override
    def on_train_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if check_main_process(True):
            self.training_bar = tqdm(
                desc="Training", total=state.train_steps, dynamic_ncols=True
            )
        self.current_step = 0

    @T.override
    def on_train_step_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if check_main_process(True):
            assert (
                self.training_bar is not None
            ), f"Training bar does not exist at step {state.step}."
            self.training_bar.update(state.step - self.current_step)
            self.current_step = state.step

    @T.override
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

    @T.override
    def on_inference_end(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @T.override
    def on_evaluate(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        if check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @T.override
    def on_predict(self, params: EngineParams, state: State, control: Signal, **kwargs):
        if check_main_process(True):
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    @T.override
    def on_log(
        self, params: EngineParams, state: State, control: Signal, logs=None, **kwargs
    ):
        if check_main_process(True) and self.training_bar is not None:
            # self.training_bar.write(str(logs))
            pass

    @T.override
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

    @T.override
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
            logger.info("Logs: %s", pformat(logs, indent=0, compact=True))
            logger.info(
                "State: %s", pformat(state.state_dict(), indent=0, compact=True)
            )
