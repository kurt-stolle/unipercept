"""
The `Engine` class is the main class to handle training and evaluation of any kind of models with any kind of datasets.
"""

from __future__ import annotations
import sys
import enum as E
import functools
import gc
import math
import multiprocessing
import operator
import os
import re
import shutil
import time
import typing as T
from datetime import datetime
from uuid import uuid4
import uuid

import torch
import torch._dynamo
import torch._dynamo.config
import torch.nn as nn
import torch.optim
import torch.types
import torch.utils.data
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image as pil_image
from tabulate import tabulate
from tensordict import TensorDict, TensorDictBase
from timm.scheduler.scheduler import Scheduler as TimmScheduler
from torch.utils.data import Dataset
from typing_extensions import override

from unipercept import file_io
import unipercept
from unipercept.data import DataLoaderFactory
from unipercept.engine._params import EngineParams, EngineStage
from unipercept.engine._trial import Trial, TrialWithParameters
from unipercept.engine.accelerate import Accelerator, find_executable_batch_size
from unipercept.engine.callbacks import CallbackType, Delegate, Event, Signal, State
from unipercept.engine.debug import DebugMode, DebugUnderflowOverflow
from unipercept.engine.memory import MemoryTracker
from unipercept.engine.writer import MemmapTensordictWriter, PersistentTensordictWriter
from unipercept.log import get_logger
from unipercept.state import (
    barrier,
    check_main_process,
    gather,
    get_process_count,
    get_process_index,
    get_total_batchsize,
    on_main_process,
)
from unipercept.utils.seed import set_seed
from unipercept.utils.status import StatusDescriptor
from unipercept.utils.tensorclass import Tensorclass
from unipercept.utils.time import ProfileAccumulator, profile
from unipercept.utils.typings import Pathable
from unipercept.utils.ulid import ULID

torch._dynamo.config.suppress_errors = True

if T.TYPE_CHECKING:
    from unipercept.evaluators import Evaluator
    from unipercept.model import ModelFactory

MaybeTensorType = T.TypeVar("MaybeTensorType", bound=torch.Tensor | None)

__all__ = ["Engine", "EngineStatus"]
_logger = get_logger(__name__)

InputType: T.TypeAlias = TensorDict | Tensorclass
TrainingOutputType: T.TypeAlias = T.Dict[str, torch.Tensor]
InferenceOutputType: T.TypeAlias = T.Sequence[T.Dict[str, T.Any]]


class EngineStatus(E.IntFlag):
    """
    The current status of the engine. This status is not part of the persistent/stored state. Used internally for
    control flow, e.g. in cases where the evaluation loop is ran while also training.
    """

    IS_TRAINING_RUN = E.auto()
    IS_EVALUATION_RUN = E.auto()
    IS_PREDICTION_RUN = E.auto()
    HP_TUNING_MODE = E.auto()
    EXPERIMENT_TRACKERS_STARTED = E.auto()
    FINDING_BATCH_SIZE = E.auto()


class FloatingPointPrecision(E.StrEnum):
    """
    Floating point precision type.
    """

    FP32 = E.auto()
    FP16 = E.auto()
    BF16 = E.auto()


class Engine:
    """
    The engine implements processes for training, evaluation, and inference.
    """

    __slots__ = (
        "_xlr",
        "_root",
        "_state",
        "_mem_tracker",
        "_params",
        "dry_run",
        "session_id",
        "_config",
        "__dict__",
    )

    def __init__(
        self,
        *,
        params: EngineParams,
        callbacks: T.Sequence[CallbackType | type[CallbackType]],
        loaders: T.Mapping[str, DataLoaderFactory],
        stages: T.Iterable[EngineStage] | None = None,
        evaluators: T.Mapping[str, T.Iterable[Evaluator]] | None = None,
        log_events: bool = False,
        dry_run: bool = False,
    ):
        self._default_setup()

        self.session_id = _generate_session_id()
        self.dry_run = dry_run

        if dry_run:
            _logger.warning("Running in dry run mode!")

        self._mem_tracker = MemoryTracker(enabled=not params.memory_tracker)
        self._mem_tracker.start("init")  # must set up as early as possible

        self._params: T.Final[EngineParams] = params
        self._seed()
        self._state = State()
        self._xlr = None
        self._root = None

        self._dataloaders: T.Final = loaders or {}
        self._stages = list(stages) if stages is not None else []
        self._evaluators: T.Final = (
            {k: list(v) for k, v in evaluators.items()}
            if evaluators is not None
            else {}
        )

        self._signal = Signal()
        self._delegate = Delegate(callbacks, verbose=log_events)

        self._flops = 0
        self._step_last_logged = -1
        self._step_last_saved = -1
        self._step_last_evaluated = -1
        self._recover_path = None  # See: `recover` method

        self._edge(Event.ON_CREATE)
        self._mem_tracker.stop_and_update_metrics("init")

    status = StatusDescriptor(EngineStatus, default=EngineStatus(0))

    @property
    def _evaluated_in_last_step(self) -> bool:
        return (self._state.step == self._step_last_evaluated) and (
            self._state.step > 0
        )

    @property
    def _saved_in_last_step(self) -> bool:
        return (self._state.step == self._step_last_saved) and (self._state.step > 0)

    @property
    def _logged_in_last_step(self) -> bool:
        return (self._state.step == self._step_last_logged) and (self._state.step > 0)

    ##############
    # Public API #
    ##############

    @override
    def __str__(self) -> str:
        args = ", \n".join(
            [
                (f"\t{k}={v}").replace("\n", "\n\t")
                for k, v in {
                    "config": str(self._params),
                    "state": str(self._state),
                    "status": tuple(map(str, self.status)),
                }.items()
            ]
        )

        return f"{self.__class__.__name__}(\n{args}\n)"

    @override
    def __repr__(self) -> str:
        return str(self)

    @property
    def xlr(self):
        if self._xlr is not None:
            return self._xlr

        xlr = Accelerator.from_engine_params(self._params, self.session_dir)
        xlr.register_for_checkpointing(self._state)

        self._xlr = xlr

        return xlr

    @property
    def session_dir(self) -> file_io.Path:
        """
        Returns the local path to the root directory of this engine as a ``pathlib.Path`` class.
        """
        if self._root is None:
            self._root = file_io.Path(
                f"//output/{self._params.project_name}/{str(self.session_id)}"
            )
            self.xlr  # Force initialization of Accelerator
        return self._root

    @session_dir.setter
    def session_dir(self, value: Pathable) -> None:
        if self._xlr is not None:
            msg = "Cannot change the root directory after the engine has started a session."
            raise RuntimeError(msg)
        self._root = file_io.Path(value)

    @property
    def config_path(self) -> file_io.Path:
        return self.session_dir / "config.yaml"

    @property
    def config_name(self) -> str:
        try:
            return self.config.get("name", "unnamed")
        except FileNotFoundError:
            return "unnamed"

    @property
    def config(self) -> dict[str, T.Any]:
        """
        Attempt to locate the configuration YAML file for the current project.
        If that does not exist, return None. If it does exist, return the configuration object.
        """
        from unipercept.config import load_config

        if self._config is not None:
            return self._config

        path = self.config_path
        if not path.exists():
            msg = f"Could not find configuration file at {path!r}"
            raise FileNotFoundError(msg)

        _logger.info("Loading configuration from %s", path)

        try:
            lazy = load_config(str(path))
        except Exception as e:  # noqa: PIE786
            msg = f"Could not load configuration from {path!r} {e}"
            _logger.warning(msg)
            return {}

        lazy_obj = OmegaConf.to_container(lazy, resolve=False)
        assert isinstance(lazy_obj, dict)
        return T.cast(dict[str, T.Any], lazy_obj)

    @config.setter
    def config(self, value: DictConfig) -> None:
        from unipercept.config import save_config

        if check_main_process():
            path = self.config_path
            if path.exists():
                msg = f"Configuration file already exists at {path}"
                raise FileExistsError(msg)
            _logger.info("Saving configuration to %s", path)
            save_config(value, str(path))
        self._config = None  # NOTE: loaded ad-hoc

    @property
    def logging_dir(self) -> file_io.Path:
        """
        Returns the local path to the logs directory of this engine as a ``pathlib.Path`` class.
        """
        return file_io.Path(self.xlr.logging_dir)

    @property
    def outputs_dir(self) -> file_io.Path:
        """
        Returns the local path to the outputs directory of this engine as a ``pathlib.Path`` class.
        """
        return file_io.Path(self.xlr.project_dir)

    @property
    def states_dir(self) -> file_io.Path:
        """
        Every stage has a unique checkpoints directory - this is because checkpoints between stages are often incompatible
        """
        assert self._state.stage >= 0, f"{self._state.stage=}"
        return self.outputs_dir / "states" / f"stage_{self._state.stage}"

    @property
    def models_dir(self) -> file_io.Path:
        assert self._state.stage >= 0, f"{self._state.stage=}"
        return self.outputs_dir / "models" / f"stage_{self._state.stage}"

    def recover(
        self,
        model: nn.Module | None = None,
        checkpoint: str | file_io.Path | None = None,
    ) -> None:
        """
        Recover a model's state and the engine's state from the given checkpoint. The model is prepared in
        evaluation mode.
        """

        if model is not None:
            self.xlr.prepare_model(model, evaluation_mode=True)

        if checkpoint is not None:
            self._recover_path = str(checkpoint)
            self._load_state(self._recover_path)  # type: ignore

        _logger.info("Recovered engine state at step %d", self._state.step)

        return None

    def run_training_procedure(
        self, model_factory, start_stage: int, *, weights: str | None = None
    ):
        """
        Run the training procedure for a specific stage. This method is called by the `train` method.
        """

        _logger.info(
            "Starting training procedure:\n%s",
            tabulate(
                [("starting stage", start_stage), ("initial weights", weights)],
                tablefmt="simple",
            ),
        )

        print("\n\n", file=sys.stderr, flush=True)
        print("\n\n", file=sys.stderr, flush=True)
        weights = weights
        stage_num = start_stage
        while True:
            weights = self.run_training(model_factory, stage=stage_num, weights=weights)
            print("\n\n", file=sys.stderr, flush=True)
            print("\n\n", file=sys.stderr, flush=True)
            stage_num += 1
            if stage_num >= len(self._stages):
                break
            _logger.info(
                "Training completed for stage %d. Moving to next...", stage_num - 1
            )

        _logger.info(
            "Training completed for all stages: \n%s",
            tabulate([("final weights", weights)], tablefmt="simple"),
        )

    @status(EngineStatus.IS_TRAINING_RUN)
    def run_training(
        self,
        model_factory: ModelFactory,
        *,
        trial: Trial | None = None,
        stage: int | EngineStage | None = None,
        weights: str | None = None,
    ) -> str:
        """
        Train a model.

        Parameters
        ----------
        model_factory
            A factory function that returns a model.
        loader_factory
            A factory function that returns a data loader.
        checkpoint
            A checkpoint to resume training from.
        trial
            The trial to train.
        stage
            The stage to train. If not specified, the current stage is used.
        weights
            Path to a checkpoint to load **model** weights from.
        """

        gc.collect()
        torch.cuda.empty_cache()
        self.xlr.free_memory()
        time.sleep(1.0)

        self._signal = Signal()

        # Memory metrics - must set up as early as possible
        self._mem_tracker.start("train")

        if stage is None:
            stage_num = self._state.stage
            assert stage_num >= 0, "Expected stage to be set"
            stage = self._stages[stage_num]
        elif isinstance(stage, int):
            if stage < 0 or stage >= len(self._stages):
                raise ValueError(
                    f"Stage {stage} is out of bounds. This engine has {len(self._stages)} stages, "
                    "and a value of -1 could indicate that the stage was recovered from a checkpoint "
                    "that used a custom StageDefinition instead of a number."
                )
            stage_num = stage
            stage = self._stages[stage]
        else:
            try:
                stage_num = self._stages.index(stage)
            except ValueError:
                stage_num = -1

        _logger.info(f"Starting training procedure for stage {stage_num}...")

        if not isinstance(stage, EngineStage):
            raise TypeError(
                f"Expected stage to be of type EngineStage, got {type(stage)}"
            )

        trial = TrialWithParameters(
            name="stage_" + str(stage_num),
            config=stage.model_config,
            weights=weights,
            parent=trial,
        )

        @find_executable_batch_size(starting_batch_size=stage.batch_size)
        def train(batch_size: int) -> nn.Module:
            """
            This inner function accepts a parameter batch_size, which is automatically
            tuned to the maximum batch size that fits into memory.

            The batch size is always less than or equal to the starting batch size,
            and for reproduction purposes, the accumulation steps are adjusted to
            emulate training at original batch size. Note that this does not guarantee
            perfect reproducibility.
            """

            # Crash when FINDING_BATCH_SIZE status is missing. This status is removed
            # after the first logging step.
            if EngineStatus.FINDING_BATCH_SIZE not in self.status:
                msg = "Aborting training (OOM)"
                raise RuntimeError(msg)

            # gradient_accumulation = stage.gradient_accumulation * (
            #    stage.batch_size // batch_size
            # )
            gradient_accumulation = 1  # PyTorch 2.2: broken
            assert (
                gradient_accumulation > 0
            ), "Expected gradient accumulation to be greater than 0"

            loader, steps_per_epoch, updates_per_epoch = self.build_training_dataloader(
                stage.dataloader, batch_size, gradient_accumulation
            )
            model = model_factory(overrides=trial.config, weights=trial.weights)
            scheduled_epochs = stage.get_epochs(steps_per_epoch)
            assert (
                scheduled_epochs > 0
            ), "Expected scheduled epochs to be greater than 0"
            optimizer = stage.optimizer(model)
            scheduler, train_epochs = stage.scheduler(
                optimizer, scheduled_epochs, updates_per_epoch
            )
            assert train_epochs > 0, "Expected train epochs to be greater than 0"

            _logger.info(
                f"Training for {train_epochs} epochs in {train_epochs*steps_per_epoch} steps"
            )

            # Reset the state
            self._state.reset()
            self._state.stage = stage_num
            self._state.logging_steps = self._params.logging_steps
            self._state.eval_steps = self._params.get_eval_interval_steps(
                steps_per_epoch
            )
            self._state.save_steps = self._params.get_save_interval_steps(
                steps_per_epoch
            )
            self._state.train_steps = stage.get_steps(steps_per_epoch)
            self._state.gradient_accumulation = gradient_accumulation
            self._state.best_metric = None
            self._state.trial_name = trial.name
            self._state.trial_params = trial.config

            return self.run_training_loop(
                loader,
                model,
                optimizer,
                scheduler,
                trial=trial,
            )

        # Add FINDING_BATCH_SIZE flag to status
        self.status |= EngineStatus.FINDING_BATCH_SIZE
        result = train()

        # Save final model weights
        last_weights = self._save_weights(None, result)

        return last_weights

    @status(EngineStatus.IS_EVALUATION_RUN)
    @torch.no_grad()
    def run_evaluation(
        self,
        model_factory: ModelFactory,
        trial: Trial | None = None,
        *,
        suites: T.Collection[str] | None = None,
        prefix: str = "evaluation",
    ) -> dict[str, float]:
        if self.dry_run:
            _logger.info("Dry run: skipping evaluation")
            return {}
        _logger.info("Starting evaluation procedure...")

        metrics_overall = {}

        for loader_key, handlers in self._evaluators.items():
            if suites is not None and loader_key not in suites:
                continue
            _logger.info(
                "Running inference on loader '%s' for %d handlers",
                loader_key,
                len(handlers),
            )

            prefix_suite = "/".join([prefix, loader_key])

            torch.cuda.empty_cache()
            gc.collect()
            # Memory metrics - must set up as early as possible
            self._mem_tracker.start("eval")

            if trial is not None:
                model = model_factory(overrides=trial.config, weights=trial.weights)
            else:
                model = model_factory()

            loader_factory = self._dataloaders[loader_key]
            loader = loader_factory(1, use_distributed=True)

            metrics, samples_processed = self.run_inference_loop(
                model, loader, prefix=prefix_suite, handlers=handlers
            )

            self._train_log(metrics)
            self._edge(Event.ON_EVALUATE, metrics=metrics)
            self._mem_tracker.stop_and_update_metrics("eval", metrics)

            del loader
            del model

            for metric_key in list(metrics.keys()):
                if not metric_key.startswith(prefix_suite):
                    metrics[prefix_suite] = metrics.pop(metric_key)

            metrics_overall.update(metrics)

        if len(metrics_overall) == 0:
            _logger.warning("No metrics were logged during evaluation")

        return metrics_overall

    @T.overload
    def build_training_dataloader(
        self,
        dataloader: str | DataLoaderFactory,
        batch_size: int,
        gradient_accumulation: None = None,
    ) -> tuple[torch.utils.data.DataLoader, int, None]:
        ...

    @T.overload
    def build_training_dataloader(
        self,
        dataloader: str | DataLoaderFactory,
        batch_size: int,
        gradient_accumulation: int,
    ) -> tuple[torch.utils.data.DataLoader, int, int]:
        ...

    def build_training_dataloader(
        self,
        dataloader: str | DataLoaderFactory,
        batch_size: int,
        gradient_accumulation: int | None = None,
    ) -> tuple[torch.utils.data.DataLoader, int, int | None]:
        """
        Build a training dataloader.

        Parameters
        ----------
        dataloader : str | DataLoaderFactory
            The key of the dataloader or a callable that returns a dataloader.
        batch_size : int
            The batch size to use for training.
        gradient_accumulation : int | None
            The number of gradient accumulation steps. When None, the amount of updates
            per epoch is not calculated.

        Returns
        -------
        torch.utils.data.DataLoader
            The training dataloader.
        int
            The number of steps per epoch.
        int | None
            The number of updates per epoch. When ``gradient_accumulation`` is None,
            this value is None.
        """

        # Divide batch size over the amount of processes
        assert batch_size % get_process_count() == 0, (
            f"Training batch size {batch_size} must be divisible over the amount of "
            f"processes {get_process_count()}."
        )

        if isinstance(dataloader, str):
            dl_factory = self._dataloaders[dataloader]
            dl = dl_factory(batch_size // get_process_count(), use_distributed=False)
        elif callable(dataloader):
            dl = dataloader(batch_size // get_process_count(), use_distributed=False)
        else:
            raise TypeError(
                f"Expected dataloader to be a string or a callable, got {type(dataloader)}"
            )

        steps_per_epoch = len(dl) // get_process_count()

        if gradient_accumulation is not None:
            updates_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation)
        else:
            updates_per_epoch = None

        # Tabulate and log the loader information
        table = {
            "Batch size": batch_size,
            "Batch count": len(dl),
            "Gradient acc.": gradient_accumulation,
            "Processes": get_process_count(),
            "Steps/Epoch": steps_per_epoch,
            "Updates/Epoch": updates_per_epoch,
        }

        _logger.debug(
            "Using dataloader settings:\n%s", tabulate(table.items(), tablefmt="simple")
        )

        return dl, steps_per_epoch, updates_per_epoch

    def run_training_step(self, model: nn.Module, inputs: InputType) -> TensorDict:
        """
        A single training step (forward + backward + update).
        """
        model.train()
        output: TensorDict = model(inputs)

        if "losses" in output.keys():
            losses: TensorDict = output["losses"]
        else:
            losses = output

        loss_tensor = torch.stack([loss for loss in losses.values()])  # type: ignore

        if self._params.train_sum_losses:
            self.xlr.backward(loss_tensor.sum())
        else:
            self.xlr.backward(loss_tensor, gradient=torch.ones_like(loss_tensor))

        losses.detach_()
        return losses.apply(lambda _l: _l / self._state.gradient_accumulation)

    def run_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: TimmScheduler,
        **kwargs,
    ) -> nn.Module:
        """
        The main training loop. This method is called by the `train` method.
        """

        # Backend configuration
        self.xlr.gradient_accumulation_steps = self._state.gradient_accumulation
        self.xlr.free_memory()
        self._start_experiment_trackers()

        # Sync backnorm
        if self._params.convert_sync_batchnorm:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = self.xlr.prepare_model(model)
        loader, scheduler, optimizer = self.xlr.prepare(loader, scheduler, optimizer)

        # First load the initial weights, then the state
        try:
            self._load_state(None)  # type: ignore
        except FileNotFoundError:
            _logger.info("Could not load state from checkpoint")

        # Debugging
        # debug_overflow = DebugUnderflowOverflow(model)  # noqa
        if DebugMode.UNDERFLOW_OVERFLOW & self._params.debug != 0:
            DebugUnderflowOverflow(model)

        # Variables that track the progress of the training
        time_start = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Create a loss tensor to avoid synchronization of TPUs through .item()
        tr_loss: TensorDict | None = None

        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0

        self._edge(Event.ON_TRAIN_BEGIN, model=model)

        total_session_samples = 0
        total_session_steps = 0

        steps_per_epoch = len(loader)
        start_epoch = math.floor(self._state.epoch)
        steps_trained_in_current_epoch = (
            self.xlr.step
        )  # int((self._state.epoch - start_epoch) * steps_per_epoch)

        assert steps_trained_in_current_epoch == int(
            (self._state.epoch - start_epoch) * steps_per_epoch
        ), (
            f"Expected {steps_trained_in_current_epoch} to be equal to "
            f"int(({self._state.epoch} - {start_epoch}) * {steps_per_epoch})"
        )

        train_epochs = math.ceil(self._state.train_steps / steps_per_epoch)

        # Check if the loader requires an epochs state
        if hasattr(loader, "epoch"):
            setattr(loader, "epoch", start_epoch)
        if hasattr(loader.sampler, "epoch"):
            setattr(loader.sampler, "epoch", start_epoch)

        if self.xlr.is_main_process:
            _logger.info(f"Starting at epoch {start_epoch} at step {self.xlr.step}")

        for epoch in range(start_epoch, train_epochs):
            # Set the epoch iterator to the original dataloader
            epoch_iterator = loader

            self._edge(Event.ON_TRAIN_EPOCH_BEGIN)

            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                _logger.debug(
                    "Skipping the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )

                epoch_iterator = self.xlr.skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0

            steps_in_epoch = len(epoch_iterator)

            for step, inputs in enumerate(epoch_iterator):
                # assert isinstance(inputs, InputType), f"Expected InputType, got {type(inputs)}"

                if total_session_samples > 0 and self.dry_run:
                    continue

                total_session_samples += 1

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % self._state.gradient_accumulation == 0:
                    self._edge(Event.ON_TRAIN_STEP_BEGIN)

                with self.xlr.accumulate(model):
                    tr_loss_step = self.run_training_step(model, inputs)

                # Add the losses individually
                if total_session_steps == 0:
                    # If this is the first step in the current session, the tensordict keys have not yet been
                    # initialized.
                    assert tr_loss is None
                    tr_loss = TensorDict(
                        {k: torch.tensor(0.0, device=v.device) for k, v in tr_loss_step.items()},  # type: ignore
                        batch_size=[],
                    )
                else:
                    assert tr_loss is not None

                for k, tr_loss_value in tr_loss.items():
                    tr_loss_step_value = tr_loss_step.get(
                        k, torch.tensor(torch.nan, device=tr_loss_step.device)
                    )
                    if self._params.logging_nan_inf_filter and (
                        torch.isnan(tr_loss_step_value)
                        or torch.isinf(tr_loss_step_value)
                    ):
                        tr_loss_value += tr_loss_value / (1 + self._state.step - self._step_last_logged)  # type: ignore
                    else:
                        tr_loss_value += tr_loss_step_value

                # Compute flops
                self._flops += float(_flops(model, inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= self._state.gradient_accumulation
                    and (step + 1) == steps_in_epoch
                )

                if (
                    total_session_samples % self._state.gradient_accumulation == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # Gradient clipping
                    if (
                        self._params.max_grad_norm is not None
                        and self._params.max_grad_norm > 0
                    ):
                        if hasattr(optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            optimizer.clip_grad_norm(self._params.max_grad_norm)  # type: ignore
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self._params.max_grad_norm)  # type: ignore
                        else:
                            self.xlr.clip_grad_norm_(
                                model.parameters(),
                                self._params.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer.step()
                    if not self.xlr.optimizer_step_was_skipped:
                        scheduler.step_update(
                            self._state.step, metric=None
                        )  # TODO metric is not used
                    else:
                        _logger.debug("Step was skipped")
                    optimizer.zero_grad()
                    self._state.step += 1
                    self._state.epoch = (
                        epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    )

                    del inputs

                    self._edge(
                        Event.ON_TRAIN_STEP_END, model=model, optimizer=optimizer
                    )
                    self._train_handle_signals(tr_loss, model, optimizer, **kwargs)
                else:
                    self._edge(Event.ON_TRAIN_SUBSTEP_END)

                total_session_steps += 1

                if self._signal.should_epoch_stop or self._signal.should_training_stop:
                    _logger.debug(
                        f"Stopping epoch @ step {step} due to signal {self._signal}"
                    )
                    break

            if tr_loss is None:
                _logger.warning("Epoch was ended without running any steps.")
                self._signal.should_training_stop = True
                tr_loss = TensorDict.from_dict({})
            else:
                assert tr_loss is not None

            scheduler.step(round(self._state.epoch), metric=None)

            self._edge(Event.ON_TRAIN_EPOCH_END)
            self._train_handle_signals(tr_loss, model, optimizer, **kwargs)

            epochs_trained += 1

            if self._signal.should_training_stop:
                break

        _logger.info("Training completed (stage %d).", self._state.stage)

        self._signal.should_save = True
        self._signal.should_evaluate = True
        self._train_handle_signals(None, model, optimizer, **kwargs)

        # Compute flops
        self._state.total_flops += self._flops
        self._flops = 0

        # Report final metrics
        metrics: dict[str, T.Any] = {}
        metrics.update(
            _build_speed_metrics(
                "training",
                time_start,
                sample_amount=total_session_samples,
                step_amount=total_session_steps,
            )
        )
        metrics["total_flops"] = self._state.total_flops

        for k in list(metrics.keys()):
            if not k.startswith("engine/"):
                metrics["engine/" + k] = metrics.pop(k)

        self._mem_tracker.stop_and_update_metrics("train", metrics)
        self._train_log(metrics)
        self._edge(Event.ON_TRAIN_END)

        return model

    def run_inference_step(
        self,
        model: nn.Module,
        inputs: TensorDict,
    ) -> TensorDictBase:
        """
        Perform an evaluation step on `model` using `inputs`.
        """

        outputs: TensorDictBase = model(inputs)
        if "predictions" in outputs.keys():
            predictions = outputs["predictions"]
        else:
            predictions = outputs

        return predictions

    @status.assert_status(
        ~(EngineStatus.IS_TRAINING_RUN | EngineStatus.IS_EVALUATION_RUN)
    )
    @status(EngineStatus.IS_PREDICTION_RUN)
    @torch.no_grad()
    def predict(
        self, model: nn.Module, datapipe: Dataset, *, prefix: str = "pred"
    ) -> TensorDict:
        raise NotImplementedError("TODO: Implement prediction")

    @torch.inference_mode()
    def run_inference_loop(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        prefix: str,
        handlers: T.Sequence[Evaluator],
        results_path: T.Optional[file_io.PathLike] = None,
        *,
        weights: T.Optional[file_io.PathLike] = None,
        batch_size: int = 1,
    ) -> tuple[dict[str, T.Any], int]:
        """
        Evaluation loop, which roughly follows the folowing procedure:

            (1) prepare the model and data loader
            (2) run prediction on the dataset and feed results through each evaluator's preprocessing function
            (3) iterate the dataset again, and run the evaluation function of each evaluator
        """

        if self.dry_run:
            _logger.info("Dry run: skipping inference")
            return {}, 0

        _logger.info("Starting inference procedure...")
        self._start_experiment_trackers(restart=False)

        torch.cuda.empty_cache()
        gc.collect()

        # Find the size of the dataset
        batch_total, batch_offsets = get_total_batchsize(dataloader, self.xlr.device)
        samples_total = batch_total * batch_size
        _logger.debug(
            f"Expecting {samples_total} samples ({batch_total} batches, offsets {batch_offsets})"
        )

        # Prepare model
        if weights is not None:
            model = self._load_weights(weights, model)

        # TODO: The FP32 wrapper added by prepare breaks our model
        if self.xlr.unwrap_model(model) is model:
            _logger.info(f"Preparing model for evaluation: {model.__class__.__name__}")
            model = self.xlr.prepare_model(model, evaluation_mode=True)
        else:
            _logger.info(
                f"Model is already prepared for evaluation: {model.__class__.__name__}"
            )

        model = self.xlr.unwrap_model(model, keep_fp32_wrapper=False)
        model.eval()

        # Global start time
        t_start_all = time.time()

        # Output memory
        if results_path is None:
            results_remove_on_exit = True
            results_path = file_io.Path(
                f"//scratch/{self._params.project_name}/{str(self.session_id)}/{prefix}-results"  # .h5"
            )
            results_path.mkdir(parents=True, exist_ok=True)
        else:
            results_remove_on_exit = False

        results_mem = MemmapTensordictWriter(
            str(results_path),
            samples_total,
            write_offset=batch_offsets[get_process_index()] * batch_size,
        )

        # print(f"writing results to {results_mem}")

        self._edge(Event.ON_INFERENCE_BEGIN, loader=dataloader)
        try:
            # Prediction
            _logger.info(f"Running inference loop on {get_process_count()} processes.")
            samples_processed = 0
            timings = ProfileAccumulator()
            for inputs in dataloader:
                with profile(timings, "copy"):
                    inputs = inputs.to(self.xlr.device, non_blocking=True)
                with profile(timings, "model"):
                    outputs = self.run_inference_step(model, inputs)
                with profile(timings, "update"):
                    samples_in_batch = inputs.batch_size[0]
                    results_merged = TensorDict(
                        {
                            "valid": torch.ones(
                                samples_in_batch,
                                dtype=torch.bool,
                                device=self.xlr.device,
                            )
                        },
                        [samples_in_batch],
                        self.xlr.device,
                    )
                    for evaluator in handlers:
                        evaluator.update(results_merged, inputs=inputs, outputs=outputs)
                with profile(timings, "write"):
                    results_mem.add(results_merged)
                    samples_processed += samples_in_batch
                with profile(timings, "event"):
                    self._edge(
                        Event.ON_INFERENCE_STEP,
                        loader=dataloader,
                        inputs=inputs,
                        outputs=outputs,
                    )

            _logger.info(
                "Profiling report on process {}/{}:\n{}".format(
                    get_process_index() + 1,
                    get_process_count(),
                    timings.to_summary().to_markdown(index=True, floatfmt=".3f"),
                )
            )
            # Sleep for 1 second appears to help with memory consistency
            time.sleep(1)
            gc.collect()
            torch.cuda.empty_cache()

            # Flush results memory
            results_mem.flush()
            barrier()

            self._edge(
                Event.ON_INFERENCE_END,
                timings=timings,
                results=results_mem,
                results_path=results_path,
            )

            # Compute metrics
            metrics: dict[str, T.Any] = {}
            if self.xlr.is_main_process:
                metrics.update(
                    _build_speed_metrics(
                        prefix,
                        t_start_all,
                        sample_amount=samples_processed,
                        step_amount=math.ceil(samples_processed * get_process_count()),
                    )
                )
                visuals: dict[str, pil_image.Image] = {}
                for evaluator in handlers or []:
                    _logger.debug(f"Running evaluation and visualization: {evaluator}")
                    metrics.update(
                        evaluator.compute(
                            results_mem.tensordict, device=self.xlr.device
                        )
                    )
                    visuals.update(evaluator.plot(results_mem.tensordict))
                self._store_visualizations(visuals, prefix=prefix)

            barrier()
        finally:
            results_mem.close()
            if self.xlr.is_main_process and results_remove_on_exit:
                _logger.info(f"Cleaning stored evaluation results at {results_path!r}")
                shutil.rmtree(results_path, ignore_errors=True)
            else:
                _logger.info("Evaluation results are stored at %s", results_path)
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{prefix}/jit_compilation_time"] = self.jit_compilation_time  # type: ignore
        _enforce_prefix(metrics, prefix)
        return metrics, samples_processed

    ############
    # Privates #
    ############

    def _seed(self) -> None:
        """
        Seed the random number generators.
        """
        set_seed(self._params.seed, fully_deterministic=self._params.full_determinism)

    def _edge(self, event: Event, **kwargs) -> None:
        """
        Called internally on every event.
        """
        self._signal = self._delegate(
            event, self._params, self._state, self._signal, **kwargs
        )

    def _start_experiment_trackers(self, *, restart: bool = True) -> None:
        """
        Initialize the experiment trackers, e.g. WandB, TensorBoard.

        Parameters
        ----------
        model : nn.Module
            The model to be watched by the loggers. Only applicable to some loggers (e.g. WandB)
        restart : bool, optional
            Whether to restart the loggers, by default True. Can be set to False to continue logging to the same
            trackers, e.g. when running an inference loop during training.

        Notes
        -----
        This should be called at the beginning of training  and inference.
        """
        if self.dry_run:
            _logger.info("Skipping experiment trackers (dry run)")
            return

        if EngineStatus.EXPERIMENT_TRACKERS_STARTED in self.status:
            if not restart:
                _logger.debug("Trackers already started, skipping setup")
                return
            else:
                _logger.info("Restarting experiment trackers")
                self.xlr.trackers.clear()
        else:
            _logger.info("Setting up experiment trackers")
            self.status |= EngineStatus.EXPERIMENT_TRACKERS_STARTED

        # Determine the job type from the status
        if self.status & EngineStatus.IS_TRAINING_RUN:
            job_type = "train"
        elif self.status & EngineStatus.IS_EVALUATION_RUN:
            job_type = "eval"
        elif self.status & EngineStatus.IS_PREDICTION_RUN:
            job_type = "pred"
        else:
            job_type = "misc"

        group_name = f"stage-{self._state.stage}" if self._state.stage >= 0 else "other"
        experiment_id = _generate_experiment_id()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Set up tracker-specific parameters
        specific_kwargs = {
            "wandb": {
                "name": self.config_name,
                "job_type": job_type,
                "reinit": False,
                "group": group_name,
                "notes": "\n\n".join(
                    (
                        self._params.notes,
                        f"Created by session: {str(self.session_id)}",
                        f"Timestamp: {timestamp}",
                    )
                ),
                "tags": list(self._params.tags),
                "id": experiment_id,
                "save_code": False,  # NOTE: Code is saved in the WandBCallback manually instead (see `wandb_integration`)
            }
        }

        # Accelerate handles the experiment trackers for us
        self.xlr.init_trackers(
            self._params.project_name,
            config=self.config,
            init_kwargs=specific_kwargs,
        )
        self._edge(
            Event.ON_TRACKERS_SETUP,
            config_path=str(self.config_path),
            session_id=str(self.session_id),
        )
        self.xlr.wait_for_everyone()

    def _stop_experiment_trackers(self) -> None:
        """
        Stop the experiment trackers. Run has been finished and cannot be logged to
        anymore.
        """
        if self.dry_run:
            _logger.info("Skipping stopping experiment trackers (dry run)")
            return

        if EngineStatus.EXPERIMENT_TRACKERS_STARTED in self.status:
            _logger.info("Stopping experiment trackers")
            for tracker in self.xlr.trackers:
                tracker.finish()
            self.xlr.trackers.clear()
            self.status &= ~EngineStatus.EXPERIMENT_TRACKERS_STARTED

    def _train_handle_signals(
        self,
        tr_loss: TensorDict | None,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        trial: Trial | None,
    ) -> None:
        """
        Called at the end of every step and epoch to log, save, and evaluate the model.
        Steps could be skipped depending on the configuration.
        """

        # SIGNAL: logging
        if self._signal.should_log and not self._logged_in_last_step:
            # Remove the FINDING_BATCH_SIZE flag from the status
            self.status &= ~EngineStatus.FINDING_BATCH_SIZE

            # Accumulate loss tracker
            assert tr_loss is not None
            logs: dict[str, float] = {}
            logs["optimizer/lr"] = _get_learning_rate(optimizer)

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = {
                loss_key: gather(loss_item).mean().item()
                for loss_key, loss_item in tr_loss.items()
            }

            # reset tr_loss to zero
            tr_loss.apply_(lambda _l: _l - _l)

            for k, v in tr_loss_scalar.items():
                logs["losses/" + k] = round(
                    v / (self._state.step - self._step_last_logged), 4
                )

            self._step_last_logged = self._state.step
            # self.store_flops()
            self._train_log(logs)

        # SIGNAL: save model
        if self._signal.should_save and not self._saved_in_last_step:
            _logger.info(
                "Saving state and model at step %d (epoch %d)",
                self._state.step,
                self._state.epoch,
            )
            state_path = self._save_state(None)
            model_path = self._save_weights(None, model)
            self._edge(Event.ON_SAVE, model_path=model_path, state_path=state_path)
            self._step_last_saved = self._state.step

        # SIGNAL: evaluate model
        if self._signal.should_evaluate and not self._evaluated_in_last_step:
            _logger.info(
                "Starting evaluation cycle @ step %d / epoch %d",
                self._state.step,
                self._state.epoch,
            )

            self.run_evaluation(lambda *args, **kwargs: model, trial=trial)
            self._step_last_evaluated = self._state.step

    def _train_log(self, logs: dict[str, T.Any]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Parameters
        ----------
        logs : dict[str, float]
            The logs to be logged.
        """
        logs["engine/epoch"] = round(self._state.epoch, 2)
        logs["engine/step"] = self._state.step
        logs["engine/epoch_step"] = self.xlr.step
        logs["engine/status"] = self.status

        self._edge(Event.ON_LOG, logs=logs)  # NOTE: logs may be updated in-place

        self._state.log_history.append(logs)
        if len(self._state.log_history) > self._params.logging_history:
            self._state.log_history.pop(0)
        self.xlr.log(logs, step=self._state.step)

    def _load_weights(self, path: Pathable, model: nn.Module) -> nn.Module:
        """
        Load the model checkpoint at the given path.

        Parameters
        ----------
        path
            The path to the model checkpoint.
        model
            The model to load the checkpoint into.

        Returns
        -------
        nn.Module
            The model with the loaded checkpoint.
        """
        from accelerate import load_checkpoint_and_dispatch

        _logger.debug("Loading weights from %s", path)
        return load_checkpoint_and_dispatch(model, file_io.get_local_path(path))

    def _save_weights(self, path: T.Optional[Pathable], model: nn.Module) -> str:
        """
        Save a model, unwrapping it from the accelerator

        Parameters
        ----------
        output_dir
            The directory to save the model checkpoints to.
        """

        path = file_io.Path(path or (self.models_dir / f"step_{self._state.step}"))

        barrier()

        if check_main_process():
            path.mkdir(exist_ok=True, parents=True)
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            self.xlr.save_model(
                self.xlr.unwrap_model(model),
                save_directory=str(path),
                safe_serialization=True,
            )
            _cleanup_generated_items(
                self.models_dir, self._params.save_total_limit or 1
            )

        barrier()

        return str(path)

    def _load_state(self, path: T.Optional[Pathable]) -> None:
        """
        Load the engine state from the given path, if no path is given, the last checkpoint is used.
        """
        if path is not None:
            path = file_io.get_local_path(path)
        elif self._recover_path is not None:
            path = file_io.get_local_path(self._recover_path)
        else:
            if (
                not file_io.isdir(self.states_dir)
                or len(file_io.ls(self.states_dir)) == 0
            ):
                raise FileNotFoundError(
                    "No engine state path given and no automatic checkpoints found."
                )
            path = _find_latest_checkpoint(self.states_dir)

        _logger.info("Loading state from %s", path)

        self.xlr.load_state(file_io.get_local_path(path))  # type: ignore

    def _save_state(self, path: T.Optional[Pathable]) -> str:
        """
        Save the engine state for recovery/resume. Sometimes called a 'checkpoint'.
        """

        path = file_io.Path(path or (self.states_dir / f"step_{self._state.step}"))

        barrier()

        self.xlr.save_state(path)  # type: ignore
        if check_main_process():
            _cleanup_generated_items(
                self.states_dir, self._params.save_total_limit or 1
            )

        barrier()

        return str(path)

    def _store_visualizations(
        self, visuals: dict[str, pil_image.Image], prefix: str
    ) -> None:
        """
        Store visualizations that are provided as a mapping of (key) -> (PIL image).
        """

        _logger.info(
            f"Storing visualizations ({len(visuals)} total): {list(visuals.keys())}"
        )

        for key, img in visuals.items():
            if self._params.eval_write_visuals:
                img_path = (
                    file_io.Path(self.xlr.project_dir)
                    / "visuals"
                    / f"{prefix}-{self._state.step}"
                    / f"{key}.png"
                )
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(img_path)

            wandb_run = self.xlr.get_tracker("wandb")
            if wandb_run is not None:
                wandb_run.log(
                    {f"{prefix}/{key}": wandb.Image(img)}, step=self._state.step
                )

    def _default_setup(self):
        """
        Sets default configuration values in third-party libraries.
        """
        # NOTE: This should no longer be nessecary since PyTorch 1.8.0
        # slurm = unipercept.integrations.slurm_integration.SLURMEnvironment()
        #
        # if slurm.is_slurm_job:
        #    N_M = slurm.cpus_per_gpu
        # else:
        #    N = multiprocessing.cpu_count()
        #    M = get_process_count()
        #    N_M = math.floor(N / M)
        # torch.set_num_threads(N_M)
        pass


def _flops(model: nn.Module, inputs: InputType) -> int:
    """
    Uses that method to compute the number of floating point
    operations for every backward + forward pass. If using another model, either implement such a method in the
    model or subclass and override this method.

    Parameters
    ----------
    inputs
        The inputs and targets of the model.

    Returns
    -------
    int
        The number of floating-point operations.
    """
    try:
        flops_fn: T.Callable[[InputType], int] = model.floating_point_ops
    except AttributeError:
        return 0
    return flops_fn(inputs)


def _get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the average learning rate of an optimizer, which is the average of the learning rates of all parameter groups.

    TODO: Should this be changed to the maximum or per-group learning rate? (@kurt-stolle)
    """
    lr_list = list(
        map(
            lambda lr: lr.item() if torch.is_tensor(lr) else float(lr),
            map(
                operator.itemgetter("lr"),
                optimizer.param_groups,
            ),
        )
    )
    return sum(lr_list) / len(lr_list)


def _build_speed_metrics(
    prefix: str,
    start_time: float,
    *,
    sample_amount: int | None = None,
    step_amount: int | None = None,
) -> dict[str, T.Any]:
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Parameters
    ----------
    prefix : str
        The prefix to use for the metric names.
    start_time : float
        The time in seconds before the operation to be measured has started.
    sample_amount : int, optional
        The number of samples processed by the operation to be measured.
    step_amount : int, optional
        The number of steps processed by the operation to be measured.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the measured metrics.
    """
    delta_time = time.time() - start_time
    result = {f"{prefix}/time": round(delta_time, 4)}
    if delta_time == 0:
        return result
    if sample_amount is not None:
        samples_per_second = sample_amount / delta_time
        result[f"{prefix}/samples_per_second"] = round(samples_per_second, 3)
    if step_amount is not None:
        steps_per_second = step_amount / delta_time
        result[f"{prefix}/steps_per_second"] = round(steps_per_second, 3)
    return result


_RE_NUMERIC_SUFFIX = re.compile(r"(\d+)$")


def _sort_children_by_suffix(path: Pathable) -> T.Iterable[str]:
    """
    Sort the children of a path by the numeric suffix.

    Parameters
    ----------
    path
        A path to some directory, containing children with numeric suffixes, i.e. item-1, item2, it3, etc.

    Yields
    ------
    str
        The path to the child.
    """
    items = file_io.ls(str(path))
    items = map(lambda p: (p, _RE_NUMERIC_SUFFIX.search(p)), items)
    items = T.cast(
        list[tuple[str, re.Match]], filter(lambda p: p[1] is not None, items)
    )
    items = sorted(items, key=lambda p: int(p[1].group(1)))

    for item, _ in items:
        item_full = file_io.join(path, item)
        yield item_full


def _find_recent_generated_item(path: Pathable) -> T.Optional[str]:
    """
    Find the most recent item in a directory with a numeric suffix.

    Parameters
    ----------
    path
        A path to some directory, containing children with numeric suffixes, i.e. item-1, item2, it3, etc.

    Returns
    -------
    str | None
        The path to the most recent child, or None if no children were found.
    """
    items = list(_sort_children_by_suffix(path))
    if not items:
        return None
    return items[-1]


def _cleanup_generated_items(path: Pathable, max_items: int) -> None:
    """
    Given some path, list all child items and sort by the suffix number, then remove all items except the last.

    E.g. for items:
    - otherkey-200
    - item-1
    - item-600
    - last-800
    - item-123

    For ``max_items=3`` we would keep the last three items: ``otherkey-200``, ``item-600``, and ``last-800``.


    Parameters
    ----------
    path
        Path to some directory.
    max_items
        Amount of items to keep.
    """

    items = list(_sort_children_by_suffix(path))

    if len(items) <= max_items:
        return

    for child in items[:-max_items]:
        if file_io.isdir(child):
            local_path = file_io.get_local_path(child)
            shutil.rmtree(local_path, ignore_errors=False)
        else:
            assert file_io.exists(child), f"Expected {child} to exist"
            file_io.rm(child)


def _enforce_prefix(metrics: dict[str, T.Any], prefix: str, sep: str = "/") -> None:
    """
    Enforce a prefix on all keys in `metrics`. This is ran in-place.
    """
    if not prefix.endswith(sep):
        prefix = prefix + sep
    for key in list(metrics.keys()):
        if key.startswith(prefix):
            continue
        metrics[prefix + key] = metrics.pop(key)


def _generate_session_id() -> str:
    """
    Generates a session ID on the main process and synchronizes it with all other processes.
    Must be called after the process group has been initialized.
    """

    from torch.distributed import broadcast_object_list, is_available, is_initialized

    from unipercept.state import check_distributed, check_main_process

    def _read_session_name():
        return str(ULID.generate())

    if check_distributed():
        if not is_available():
            msg = "Distributed training is not available."
            raise RuntimeError(msg)

        if not is_initialized():
            msg = "Distributed training is not initialized."
            raise RuntimeError(msg)

        name_list = [_read_session_name() if check_main_process(local=False) else None]

        broadcast_object_list(name_list)

        name = name_list[0]
        assert name is not None, "No name was broadcast"
        return name
    else:
        return _read_session_name()


def _generate_experiment_id() -> str:
    """
    Generate a unique ID for the experiment.
    """
    import wandb.util

    return str(wandb.util.generate_id(length=8))
