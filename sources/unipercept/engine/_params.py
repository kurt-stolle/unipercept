from __future__ import annotations

import dataclasses as D
import enum as E
import logging
import typing as T

from typing_extensions import override

from unipercept.data import DataLoaderFactory
from unipercept.engine._optimizer import OptimizerFactory
from unipercept.engine._scheduler import SchedulerFactory
from unipercept.engine.debug import DebugMode
from unipercept.log import LOG_LEVELS, get_logger
from unipercept.state import check_main_process

__all__ = ["EngineParams", "EngineStage", "InferencePrecision", "Interval"]

_logger = get_logger(__name__)

_DEFAULT_EXPERIMENT_TRACKERS: set[str] = {"tensorboard"}


class InferencePrecision(E.StrEnum):
    """
    Defines the different modes of FP16 inference.
    """

    DEFAULT = E.auto()
    FULL_FP16 = E.auto()
    FULL_BF16 = E.auto()


class Interval(T.NamedTuple):
    """
    The engine runs on intervals of steps, which can be defined in terms of epochs.

    Traditionally the epoch is defined as the amount of steps to be trained such
    that the model has 'seen' the full dataset. This is however not always the case or
    results in vague definitions, e.g. in the case of random clipping or
    infinite data sources.

    We recommend interpreting the ``steps_per_epoch`` as a
    hyperparameter that defines when the model has seen the scope of the dataset,
    i.e. all classes have been seen in most of the possible contexts.
    """

    amount: int
    unit: T.Literal["steps", "epochs"]

    def get_steps(self, steps_per_epoch: int) -> int:
        if self.unit == "steps":
            return self.amount
        if self.unit == "epochs":
            return self.amount * steps_per_epoch

        msg = f"Unknown unit {self.unit}"
        raise ValueError(msg)

    def get_epochs(self, steps_per_epoch: int) -> float:
        if self.unit == "steps":
            return self.amount // steps_per_epoch
        if self.unit == "epochs":
            return self.amount

        msg = f"Unknown unit {self.unit}"
        raise ValueError(msg)


@D.dataclass(frozen=True, slots=True)
class EngineStage:
    """
    Defines a stage in the training process.
    """

    dataloader: str | DataLoaderFactory  # key in dataloaders dict or a factory function
    batch_size: int
    optimizer: OptimizerFactory
    scheduler: SchedulerFactory
    iterations: Interval = D.field(default=Interval(1, "epochs"), metadata={})
    gradient_accumulation: int = 1
    model_config: T.Dict[str, T.Any] = D.field(
        default_factory=dict,
        metadata={"help": "Model configuration overrides, dict of keys and values."},
    )

    def get_steps(self, steps_per_epoch: int) -> int:
        amount, unit = self.iterations

        if unit == "steps":
            return int(amount)
        elif unit == "epochs":
            return int(amount * steps_per_epoch)
        else:
            raise ValueError(f"Unknown unit {unit}")

    def get_epochs(self, steps_per_epoch: int) -> float:
        amount, unit = self.iterations

        if unit == "steps":
            value = float(amount / steps_per_epoch)
        elif unit == "epochs":
            value = float(amount)
        else:
            raise ValueError(f"Unknown unit {unit}")
        return value


@D.dataclass(match_args=False, kw_only=True)
class EngineParams:
    """
    Defines the (hyper)parameters of the engine.
    """

    project_name: str = D.field(
        default="default", metadata={"help": "Name of the project."}
    )
    notes: str = D.field(
        default="", metadata={"help": "Notes to use for the experiment."}
    )
    tags: T.Sequence[str] = D.field(
        default_factory=list, metadata={"help": "Tags to use for the experiment."}
    )

    full_determinism: bool = False
    seed: int = 42
    max_grad_norm: float = D.field(
        default=15.0, metadata={"help": "Max gradient norm."}
    )

    # Memory tracker
    memory_tracker: bool = D.field(
        default=False, metadata={"help": "Whether to track memory usage."}
    )

    # Experiment trackers
    trackers: set[str] = D.field(
        default_factory=lambda: _DEFAULT_EXPERIMENT_TRACKERS,
        metadata={"help": "Experiment trackers to use."},
    )

    # FP16 modes during inference
    inference_precision: InferencePrecision = InferencePrecision.DEFAULT

    # Convert BatchNorm to SyncBatchNorm?
    convert_sync_batchnorm: bool = D.field(
        default=False,
        metadata={"help": "Whether to convert BatchNorm to SyncBatchNorm."},
    )

    ########################################
    # Training
    ########################################

    find_unused_parameters: bool = D.field(
        default=False,
        metadata={
            "help": "When using distributed training, whether to use the `find_unused_parameters` flag in the DDP wrapper."
        },
    )

    train_sum_losses: bool = D.field(
        default=False,
        metadata={
            "help": "Whether to sum the losses instead of directly passing them to backward."
        },
    )

    ########################################
    # Evaluation
    ########################################

    eval_interval: Interval = D.field(default=Interval(1, "epochs"), metadata={})
    eval_write_visuals: bool = D.field(
        default=False,
        metadata={
            "help": (
                "Whether to save visuals during evaluation. If `True`, the visuals will be saved in the "
                "`visuals` directory of the project."
            )
        },
    )
    eval_delay: int = 0  # steps

    def get_eval_interval_steps(self, steps_per_epoch: int) -> int | None:
        """
        Get the evaluation interval in steps. If the given amount of steps is less than 1, it is interpreted
        as the ratio to the training steps (e.g., 0.1 means 10% of the total training steps).
        """

        amount, unit = self.eval_interval
        if amount <= 0:
            return None
        if unit == "steps":
            return amount
        elif unit == "epochs":
            return amount * steps_per_epoch
        else:
            raise ValueError(f"Invalid evaluation interval unit: {unit}")

    eval_accumulation_steps: T.Optional[int] = D.field(
        default=None,
        metadata={
            "help": "Number of predictions steps to accumulate before moving the tensors to the CPU."
        },
    )

    ########################################
    # Logging
    ########################################
    log_level: str = D.field(
        default="debug",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": LOG_LEVELS.keys(),
        },
    )
    log_level_replica: str = D.field(
        default="warning",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": LOG_LEVELS.keys(),
        },
    )
    log_on_each_node: bool = D.field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    logging_first_step: bool = D.field(
        default=False, metadata={"help": "Log the first global_step"}
    )
    logging_steps: int = D.field(
        default=100,
        metadata={"help": ("Log every X training steps.")},
    )
    logging_history: int = D.field(
        default=10, metadata={"help": "Number of past logs to keep in the state."}
    )
    logging_nan_inf_filter: bool = D.field(
        default=True, metadata={"help": "Filter nan and inf losses for logging."}
    )

    #################################################
    # Saving
    #################################################

    save_interval: Interval = D.field(default=Interval(1, "epochs"), metadata={})

    def get_save_interval_steps(self, steps_per_epoch: int) -> int | None:
        """
        Get the save interval in steps. If the given amount of steps is less than 1, it is interpreted
        as the ratio to the training steps (e.g., 0.1 means 10% of the total training steps).
        """

        amount, unit = self.save_interval
        if amount <= 0:
            return None
        if unit == "steps":
            return amount
        elif unit == "epochs":
            return amount * steps_per_epoch
        else:
            raise ValueError(f"Invalid save interval unit: {unit}")

    save_total_limit: T.Optional[int] = D.field(
        default=1,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    save_on_each_node: bool = D.field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )

    ###########################
    # Optimizations and JIT
    ###########################

    jit_mode_eval: bool = D.field(
        default=False,
        metadata={"help": "Whether or not to use PyTorch jit trace for inference"},
    )

    ##############################
    # Debugging and profiling
    ###############################
    debug: DebugMode = D.field(
        default=DebugMode.NONE,
        metadata={
            "help": (
                "Configures the debugging mode, see the `DebugMode` enum for more information. Defaults to `NONE`."
            )
        },
    )

    ###############################################
    # Misc
    ###############################################

    interactive: T.Optional[bool] = D.field(
        default=None,
        metadata={"help": "Whether or not to disable the tqdm progress bars."},
    )

    ###############################################
    # Target metric
    ###############################################

    metric: T.Optional[str] = D.field(
        default=None,
        metadata={
            "help": "The metric to use to compare two different models. Should be a key of the evaluation output."
        },
    )
    metric_maximize: T.Optional[bool] = D.field(
        default=None,
        metadata={"help": "Whether the `metric` should be maximized or not."},
    )

    ###############################################
    # Post-initialization defaults and sanitization
    ###############################################

    def __setup_interactive(self, **kwds) -> None:
        if self.interactive is None:
            interactive = _logger.getEffectiveLevel() > logging.WARN
        else:
            interactive = self.interactive
        self.interactive = interactive

    def __post_init__(self, **init_vars):
        self.__setup_interactive(**init_vars)

    ################################################
    # Representation and state
    ################################################

    def state_dict(self) -> dict:
        return D.asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        for k, v in state_dict.items():
            super().__setattr__(k, v)

    @override
    def __repr__(self):
        return f"Config({self.project_name=}, {self.config_name=})"

    @override
    def __str__(self):
        state_dict = self.state_dict()
        state_str = ", ".join(f"{k!r}={v!r}" for k, v in state_dict.items())
        return f"Config({state_str})"

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        if self.log_on_each_node:
            return check_main_process(local=True)
        else:
            return check_main_process(local=False)

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        if self.save_on_each_node:
            return check_main_process(local=True)
        else:
            return check_main_process(local=False)
