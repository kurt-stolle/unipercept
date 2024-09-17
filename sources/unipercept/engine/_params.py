from __future__ import annotations

import dataclasses as D
import datetime
import enum as E
import logging
import typing as T

import regex as re
import torch.compiler

from unipercept.config.env import get_env
from unipercept.data import DataLoaderFactory
from unipercept.engine.debug import DebugMode
from unipercept.log import LOG_LEVELS, get_logger
from unipercept.state import check_main_process

from ._optimizer import OptimizerFactory
from ._scheduler import SchedulerFactory
from ._types import Interval

if T.TYPE_CHECKING:
    import unipercept as up

__all__ = [
    "EngineParams",
    "InferencePrecision",
    "Interval",
    "TrainingStage",
    "EvaluationSuite",
]
_logger = get_logger(__name__)

_DEFAULT_EXPERIMENT_TRACKERS: set[str] = {"tensorboard"}


class InferencePrecision(E.StrEnum):
    """
    Defines the different modes of inference precision.
    """

    DEFAULT = E.auto()
    FULL_FP16 = "fp16"
    FULL_BF16 = "bf16"


class TrainingPrecision(E.StrEnum):
    """
    Defines the different modes of training (AMP) precision
    """

    DEFAULT = E.auto()
    AMP_FP16 = "fp16"
    AMP_BF16 = "bf16"
    AMP_INT8 = "int8"


@D.dataclass(kw_only=True)
class EvaluationSuite:
    """
    A suite of evaluators to run on a specific dataloader.
    """

    name: str = D.field(
        metadata={
            "help": (
                "The name this suite appears presents itself as. "
                "No uniqueness constraints."
            ),
            "pattern": re.compile(r"^[a-z0-9/-]+$"),
        }
    )
    loader: DataLoaderFactory
    enabled: bool = D.field(
        default=True, metadata={"help": "Whether the evaluation suite is enabled."}
    )
    batch_size: int = D.field(
        default_factory=lambda: get_env(
            int, "UP_ENGINE_EVALUATION_BATCH_SIZE", default=1
        ),
        metadata={"help": "Batch size to use during evaluation."},
    )
    handlers: dict[str, up.evaluators.Evaluator] = D.field(
        metadata={"help": "Evaluators to run on the dataset."},
    )

    def __post_init__(self):
        # Check whether the name consists only of lowercase letters, numbers, dashes and forward slashes
        name_field = EvaluationSuite.__dataclass_fields__["name"]
        name_pattern = name_field.metadata["pattern"]
        if not name_pattern.match(self.name):
            msg = (
                f"Invalid evaluation suite name '{self.name}', ensure the name "
                "consists of lowercase letters, numbers, dashes and forward slashes."
            )
            raise ValueError(msg)

        if all(h.enabled is False for h in self.handlers.values()):
            _logger.info(
                f"Disabling evaluation suite {self.name} as all handlers are disabled."
            )
            self.enabled = False


class DDPKWArgs(T.TypedDict, total=False):
    """
    A dataclass field that is used to pass additional arguments to the DDP constructor.
    """

    find_unused_parameters: bool
    broadcast_buffers: bool
    gradient_as_bucket_view: bool
    static_graph: bool


class InitProcessGroupKWArgs(T.TypedDict, total=False):
    backend: str
    init_method: str
    timedelta: datetime.timedelta


@D.dataclass(frozen=True, slots=True)
class TrainingStage:
    """
    Defines a stage in the training process.
    """

    loader: DataLoaderFactory
    batch_size: int
    optimizer: OptimizerFactory
    scheduler: SchedulerFactory | None = D.field(
        default=None,
        metadata={"help": "Scheduler to use. If None, no scheduler is used."},
    )
    iterations: Interval = D.field(default=Interval(1, "epochs"), metadata={})
    gradient_accumulation: int = 1
    model_config: dict[str, T.Any] = D.field(
        default_factory=dict,
        metadata={"help": "Model configuration overrides, dict of keys and values."},
    )

    def get_steps(self, steps_per_epoch: int) -> int:
        amount, unit = self.iterations

        if unit == "steps":
            return int(amount)
        if unit == "epochs":
            return int(amount * steps_per_epoch)
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
    memory_tracker: bool = D.field(
        default=False, metadata={"help": "Whether to track memory usage."}
    )
    trackers: set[str] = D.field(
        default_factory=lambda: _DEFAULT_EXPERIMENT_TRACKERS,
        metadata={"help": "Experiment trackers to use."},
    )
    inference_precision: InferencePrecision = InferencePrecision.DEFAULT
    training_precision: TrainingPrecision = TrainingPrecision.DEFAULT
    convert_sync_batchnorm: bool = D.field(
        default=True,
        metadata={"help": "Whether to convert BatchNorm to SyncBatchNorm."},
    )

    ########################################
    # Training
    ########################################

    ddp_kwargs: DDPKWArgs = D.field(
        default_factory=dict,
        metadata={"help": "Additional arguments to pass to the DDP constructor."},
    )

    process_group_kwargs: InitProcessGroupKWArgs = D.field(
        default_factory=dict,
        metadata={
            "help": "Additional arguments to pass to process group initialization."
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
        if unit == "epochs":
            return amount * steps_per_epoch
        raise ValueError(f"Invalid evaluation interval unit: {unit}")

    eval_accumulation_steps: int | None = D.field(
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
        if unit == "epochs":
            return amount * steps_per_epoch
        raise ValueError(f"Invalid save interval unit: {unit}")

    save_total_limit: int | None = D.field(
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

    compiler_backend: str | None = D.field(
        default=None,
        metadata={
            "help": f"The backend to use for the compiler. Available are: {torch.compiler.list_backends()}",
        },
    )
    compiler_mode: str = D.field(
        default="default",
        metadata={
            "help": "The mode to use for the compiler. See the PyTorch documentation for more information."
        },
    )
    compiler_fullgraph: bool = D.field(
        default=False,
        metadata={
            "help": "Whether or not to use the full graph for the compiler. See the PyTorch documentation for more information."
        },
    )
    compiler_dynamic: bool = D.field(
        default=False,
        metadata={
            "help": "Whether or not to use the dynamic compiler. See the PyTorch documentation for more information."
        },
    )
    compiler_options: dict[str, T.Any] = D.field(
        default_factory=dict,
        metadata={"help": "Additional options to pass to the compiler."},
    )
    compiler_optimize_ddp: bool = D.field(
        default=True,
        metadata={
            "help": "Whether or not to optimize the DDP model for the compiler. See the PyTorch documentation for more information."
        },
    )
    compiler_suppress_errors: bool = D.field(
        default=False,
        metadata={
            "help": "Whether or not to suppress errors during compilation. See the PyTorch documentation for more information."
        },
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

    interactive: bool | None = D.field(
        default=None,
        metadata={"help": "Whether or not to disable the tqdm progress bars."},
    )

    ###############################################
    # Target metric
    ###############################################

    metric: str | None = D.field(
        default=None,
        metadata={
            "help": "The metric to use to compare two different models. Should be a key of the evaluation output."
        },
    )
    metric_maximize: bool | None = D.field(
        default=None,
        metadata={"help": "Whether the `metric` should be maximized or not."},
    )

    ################################################
    # Backwards compatibility & deprecation warnings
    ################################################
    max_grad_norm: D.InitVar[float | None] = D.field(
        default=None, metadata={"help": "No-op (DEPRECATED)"}
    )
    max_grad_value: D.InitVar[float | None] = D.field(
        default=None, metadata={"help": "No-op (DEPRECATED)"}
    )

    ###############################################
    # Post-initialization defaults & sanitization
    ###############################################

    def __setup_interactive(self, **kwds) -> None:
        if self.interactive is None:
            interactive = _logger.getEffectiveLevel() > logging.WARN
        else:
            interactive = self.interactive
        self.interactive = interactive

    def __post_init__(self, max_grad_norm=None, max_grad_value=None, **init_vars):
        self.__setup_interactive(**init_vars)

        if max_grad_norm is not None or max_grad_value is not None:
            # Note: BW compat is not injected due to the complexity involved in
            #       handling the injection of callbacks at the current scope.
            _logger.warning(
                "The `max_grad_norm` and `max_grad_value` parameters are deprecated "
                "and will be removed in the next version. "
                "Use the `callbacks.GradientClippingCallback` instead. "
                "No backwards compatability is injected for these parameters!"
            )

    ################################################
    # Representation and state
    ################################################

    def state_dict(self) -> dict:
        return D.asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        for k, v in state_dict.items():
            super().__setattr__(k, v)

    # @override
    # def __str__(self):
    #     state_dict = self.state_dict()
    #     state_str = ", ".join(f"{k!r}={v!r}" for k, v in state_dict.items())
    #     return f"Config({state_str})"

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        if self.log_on_each_node:
            return check_main_process(local=True)
        return check_main_process(local=False)

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        if self.save_on_each_node:
            return check_main_process(local=True)
        return check_main_process(local=False)
