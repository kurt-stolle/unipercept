from __future__ import annotations

import enum
import enum as E
import functools
import logging
import os
import re
import typing as T
import warnings
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path

import accelerate
import unicore.file_io
from matplotlib.pyplot import step
from typing_extensions import override

from unipercept.utils.logutils import LOG_LEVELS, get_logger
from unipercept.utils.state import (
    check_main_process,
    get_process_index,
    local_main_process_first,
)
from unipercept.utils.time import get_timestamp

from .debug import DebugMode

_logger = get_logger(__name__)


_T = T.TypeVar("_T")


_DEFAULT_EXPERIMENT_TRACKERS: set[str] = {"wandb"}


class InferencePrecision(E.StrEnum):
    """
    Defines the different modes of FP16 inference.
    """

    DEFAULT = E.auto()
    FULL_FP16 = E.auto()
    FULL_BF16 = E.auto()


@dataclass(slots=True, unsafe_hash=True, match_args=False, kw_only=True, weakref_slot=True)
class TrainConfig:
    project_name: str
    session_name: str = field(default_factory=get_timestamp)
    root: str = "//output/{project_name}/{session_name}"

    train_batch_size: int = 8
    infer_batch_size: int = 1
    full_determinism: bool = False
    seed: int = 42

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    max_grad_norm: float = field(default=5.0, metadata={"help": "Max gradient norm."})

    # Memory tracker
    memory_tracker: bool = field(default=False, metadata={"help": "Whether to track memory usage."})

    # Experiment trackers
    trackers: set[str] = field(
        default_factory=lambda: _DEFAULT_EXPERIMENT_TRACKERS, metadata={"help": "Experiment trackers to use."}
    )

    # FP16 modes during inference
    inference_precision: InferencePrecision = InferencePrecision.DEFAULT

    ########################################
    # Training
    ########################################

    find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "When using distributed training, whether to use the `find_unused_parameters` flag in the DDP wrapper."
        },
    )

    train_sum_losses: bool = field(
        default=False, metadata={"help": "Whether to sum the losses instead of directly passing them to backward."}
    )

    train_steps: int | None = field(
        default=None,
        metadata={
            "help": "If > 0: set total number of training steps to perform. Mutually exclusive with `train_epochs`."
        },
    )
    train_epochs: int | None = field(
        default=None,
        metadata={
            "help": "If > 0: set total number of training epochs to perform. Mutually exclusive with `train_steps`."
        },
    )

    def get_train_steps(self, steps_per_epoch: int) -> int:
        if (self.train_steps is not None) and (self.train_epochs is not None):
            raise ValueError("Only one of `train_steps` and `train_epochs` can be set.")
        elif self.train_steps is not None:
            return self.train_steps
        elif self.train_epochs is not None:
            return self.train_epochs * steps_per_epoch
        else:
            raise ValueError("Either `train_steps` or `train_epochs` must be greater than zero.")

    def get_train_epochs(self, steps_per_epoch: int) -> int:
        if (self.train_steps is not None) and (self.train_epochs is not None):
            raise ValueError(
                f"Only one of `train_steps` and `train_epochs` can be set. Got {self.train_steps=} and {self.train_epochs=}."
            )
        elif self.train_epochs is not None:
            return self.train_epochs
        elif self.train_steps is not None:
            epochs = self.train_steps // steps_per_epoch
            _logger.debug(f"Converted {self.train_steps} steps to {epochs} epochs.")
            return epochs
        else:
            raise ValueError("Either `train_steps` or `train_epochs` must be greater than zero.")

    ########################################
    # Evaluation
    ########################################

    eval_steps: T.Optional[float | int] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
                "Mutually exclusive with `eval_epochs`."
            )
        },
    )
    eval_epochs: T.Optional[float | int] = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X epochs. Should be an integer or a float in range `[0,1)`. "
                "Mutually exclusive with `eval_steps`."
            )
        },
    )
    eval_write_visuals: bool = field(
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

        if (self.eval_steps is not None) and (self.eval_epochs is not None):
            raise ValueError("Only one of `eval_steps` and `eval_epochs` can be set.")
        elif self.eval_steps is not None:
            steps = self.eval_steps
        elif self.eval_epochs is not None:
            steps = self.eval_epochs * steps_per_epoch
        else:
            raise ValueError("Either `eval_steps` or `eval_epochs` must be greater than zero.")

        if steps == 0:
            return None
        elif steps <= 1:
            return int(steps * self.get_train_steps(steps_per_epoch))
        elif isinstance(steps, int) or steps.is_integer():
            return int(steps)
        else:
            raise ValueError(f"Invalid value for `eval_steps` or `eval_epochs`: {steps}.")

    def get_eval_interval_epochs(self, steps_per_epoch: int) -> int | None:
        """
        Get the number of evaluation epochs to perform. If the amount of epochs is less than 1, it is interpreted
        and the ratio to the training epochs (e.g., 0.1 means 10% of the total training epochs).
        """

        # TODO: Remove this feature - translating steps to epochs may cause information loss
        warnings.warn(
            "`get_eval_interval_epochs` is deprecated and will be removed in a future version. Use `get_save_interval_steps` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if (self.eval_steps is not None) and (self.eval_epochs is not None):
            raise ValueError("Only one of `eval_steps` and `eval_epochs` can be set.")
        elif self.eval_epochs is not None:
            epochs = self.eval_epochs
        elif self.eval_steps is not None:
            epochs = self.eval_steps // steps_per_epoch
        else:
            raise ValueError("Either `eval_steps` or `eval_epochs` must be greater than zero.")

        if epochs == 0:
            return None
        elif epochs <= 1:
            return int(epochs * self.get_train_epochs(steps_per_epoch))
        elif isinstance(epochs, int) or epochs.is_integer():
            return int(epochs)
        else:
            raise ValueError(f"Invalid value for `eval_steps` or `eval_epochs`: {epochs}.")

    eval_accumulation_steps: T.Optional[int] = field(
        default=None,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )

    ########################################
    # Logging
    ########################################
    log_level: str = field(
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
    log_level_replica: str = field(
        default="warning",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": LOG_LEVELS.keys(),
        },
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
        },
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: int = field(
        default=100,
        metadata={"help": ("Log every X training steps.")},
    )
    logging_history: int = field(default=10, metadata={"help": "Number of past logs to keep in the state."})
    logging_nan_inf_filter: bool = field(default=True, metadata={"help": "Filter nan and inf losses for logging."})

    #################################################
    # Saving
    #################################################

    save_steps: float | None | int = field(
        default=None,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`."
                "If smaller than 1, will be interpreted as ratio of total training steps."
                "Mutually exclusive with `save_epochs`."
            )
        },
    )
    save_epochs: float | None | int = field(
        default=None,
        metadata={
            "help": (
                "Save checkpoint every X epochs. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps. "
                "Mutually exclusive with `save_steps`."
            )
        },
    )

    def get_save_interval_steps(self, steps_per_epoch: int) -> int | None:
        """
        Get the save interval in steps. If the given amount of steps is less than 1, it is interpreted
        as the ratio to the training steps (e.g., 0.1 means 10% of the total training steps).
        """

        if (self.save_steps is not None) and (self.save_epochs is not None):
            raise ValueError("Only one of `save_steps` and `save_epochs` can be set.")
        elif self.save_steps is not None:
            steps = self.save_steps
        elif self.save_epochs is not None:
            steps = self.save_epochs * steps_per_epoch
        else:
            raise ValueError("Either `save_steps` or `save_epochs` must be greater than zero.")

        if steps == 0:
            return None
        elif steps <= 1:
            return int(steps * self.get_train_steps(steps_per_epoch))
        elif isinstance(steps, int) or steps.is_integer():
            return int(steps)
        else:
            raise ValueError(f"Invalid value for `save_steps` or `save_epochs`: {steps}.")

    def get_save_interval_epochs(self, steps_per_epoch: int) -> int | None:
        """
        Get the number of save epochs to perform. If the amount of epochs is less than 1, it is interpreted
        and the ratio to the training epochs (e.g., 0.1 means 10% of the total training epochs).
        """

        # TODO: Remove this feature - translating steps to epochs may cause information loss
        warnings.warn(
            "`get_save_interval_epochs` is deprecated and will be removed in a future version. Use `get_save_interval_steps` instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        if (self.save_steps is not None) and (self.save_epochs is not None):
            raise ValueError("Only one of `save_steps` and `save_epochs` can be set.")
        elif self.save_epochs is not None:
            epochs = self.save_epochs
        elif self.save_steps is not None:
            epochs = self.save_steps // steps_per_epoch
        else:
            raise ValueError("Either `save_steps` or `save_epochs` must be greater than zero.")

        if epochs == 0:
            return None
        elif epochs <= 1:
            return int(epochs * self.get_train_epochs(steps_per_epoch))
        elif isinstance(epochs, int) or epochs.is_integer():
            return int(epochs)
        else:
            raise ValueError(f"Invalid value for `save_steps` or `save_epochs`: {epochs}.")

    save_total_limit: T.Optional[int] = field(
        default=10,
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
    save_on_each_node: bool = field(
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

    jit_mode_eval: bool = field(
        default=False, metadata={"help": "Whether or not to use PyTorch jit trace for inference"}
    )

    ##############################
    # Debugging and profiling
    ###############################
    debug: DebugMode = field(
        default=DebugMode.NONE,
        metadata={
            "help": (
                "Whether or not to enable debug mode. Current options: "
                "`underflow_overflow` (Detect underflow and overflow in activations and weights), "
                "`tpu_metrics_debug` (print debug metrics on TPU)."
            )
        },
    )

    ###############################################
    # Online temporal ensembling
    ###############################################
    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    ###############################################
    # Misc
    ###############################################

    interactive: T.Optional[bool] = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    ###############################################
    # Target metric
    ###############################################

    metric: T.Optional[str] = field(
        default=None,
        metadata={
            "help": "The metric to use to compare two different models. Should be a key of the evaluation output."
        },
    )
    metric_maximize: T.Optional[bool] = field(
        default=None, metadata={"help": "Whether the `metric` should be maximized or not."}
    )

    ###############################################
    # Post-initialization defaults and sanitization
    ###############################################

    def __setup_root(self, **kwds) -> None:
        root = self.root.format(project_name=self.project_name, session_name=self.session_name)
        root = unicore.file_io.get_local_path(root)
        self.root = root

    def __setup_interactive(self, **kwds) -> None:
        if self.interactive is None:
            interactive = _logger.getEffectiveLevel() > logging.WARN
        else:
            interactive = self.interactive
        self.interactive = interactive

    def __post_init__(self, **init_vars):
        self.__setup_root(**init_vars)
        self.__setup_interactive(**init_vars)

    ################################################
    # Representation and state
    ################################################

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state_dict: dict) -> None:
        for k, v in state_dict.items():
            super().__setattr__(k, v)

    @override
    def __repr__(self):
        return f"Config({self.project_name=}, {self.session_name=})"

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

    # def get_process_log_level(self):
    #     """
    #     Returns the log level to be used depending on whether this process is the main process of node 0, main process
    #     of node non-0, or a non-main process.

    #     For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
    #     anything) unless overridden by `log_level` argument.

    #     For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
    #     argument.

    #     The choice between the main and replica process settings is made according to the return value of `should_log`.
    #     """

    #     # convert to int
    #     log_level = LOG_LEVELS.get(self.log_level, -1)
    #     log_level_replica = LOG_LEVELS.get(self.log_level_replica, -1)

    #     log_level_main_node = logging.get_verbosity() if log_level == -1 else log_level
    #     log_level_replica_node = logging.get_verbosity() if log_level_replica == -1 else log_level_replica
    #     return log_level_main_node if self.should_log else log_level_replica_node


class ParallelMode(enum.StrEnum):
    NOT_PARALLEL = enum.auto()
    NOT_DISTRIBUTED = enum.auto()
    DISTRIBUTED = enum.auto()
