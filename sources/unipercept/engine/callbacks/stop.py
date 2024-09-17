from unipercept.log import logger

from .._params import EngineParams
from ._base import CallbackDispatcher, Signal, State


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
            logger.warning(
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
            logger.info(
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
        early_stopping_threshold: float | None = 0.0,
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
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(params, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
