"""
The `Engine` class is the main class to handle training and evaluation of any kind of models with any kind of datasets.
"""

from __future__ import annotations

import enum as E
import functools
import gc
import math
import multiprocessing
import operator
import os
import shutil
import time
import typing as T

import accelerate
import accelerate.utils
import torch
import torch._dynamo
import torch._dynamo.config
import torch.nn as nn
import torch.optim
import torch.utils.data
import wandb
from omegaconf import OmegaConf
from PIL import Image as pil_image
from tensordict import TensorDict, TensorDictBase
from timm.scheduler.scheduler import Scheduler as TimmScheduler
from torch.utils.data import Dataset
from typing_extensions import override
from unicore import file_io
from unicore.utils.status import StatusDescriptor
from unicore.utils.tensorclass import Tensorclass

import unipercept.integrations.slurm_integration
from unipercept.engine.callbacks import CallbackType, Delegate, Event, Signal, State
from unipercept.engine.debug import DebugMode, DebugUnderflowOverflow
from unipercept.engine.memory import MemoryTracker
from unipercept.engine.writer import PersistentTensordictWriter
from unipercept.log import get_logger
from unipercept.state import check_main_process, on_main_process
from unipercept.utils.seed import set_seed
from unipercept.utils.time import ProfileAccumulator, profile

from ._optimizer import OptimizerFactory
from ._params import EngineParams, InferencePrecision
from ._scheduler import SchedulerFactory
from ._trial import Trial
from ._types import DataLoaderFactory, ModelFactory

torch._dynamo.config.suppress_errors = True

if T.TYPE_CHECKING:
    from unipercept.evaluators import Evaluator

    try:
        from wandb.sdk.wandb_run import Run as WandBRun
    except ImportError:
        pass

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


class Engine:
    """
    The engine implements processes for training, evaluation, and inference.
    """

    __slots__ = [
        "_params",
        "_xlr",
        "_state",
        "_stages",
        "_evaluators",
        "_signal",
        "_delegate",
        "_get_optimizer",
        "_get_scheduler",
        "_loaders",
        "_mem_tracker",
        "_flops",
        "_globalstep_last_logged",
        "_recover_path",
        "_notes",
        "_tags",
        "__dict__",
    ]

    def __init__(
        self,
        *,
        params: EngineParams,
        optimizer: OptimizerFactory,
        scheduler: SchedulerFactory,
        callbacks: T.Sequence[CallbackType | type[CallbackType]],
        loaders: T.Mapping[str, DataLoaderFactory],
        stages: T.Iterable[str] | None = None,
        evaluators: T.Mapping[str, T.Iterable[Evaluator]] | None = None,
        log_events: bool = False,
    ):
        self._mem_tracker = MemoryTracker(enabled=not params.memory_tracker)
        self._mem_tracker.start("init")  # must set up as early as possible

        _logger.info("Initializing Engine: %s @ %s", params.project_name, params.session_name)

        self._params: T.Final[EngineParams] = params
        self._xlr = _build_accelerator(params)
        self._state = State()

        self._loaders: T.Final = loaders or {}
        self._stages: T.Final = list(stages) if stages is not None else []
        self._evaluators: T.Final = {k: list(v) for k, v in evaluators.items()} if evaluators is not None else {}

        self._default_setup()
        self._seed()

        self._signal = Signal()
        self._delegate = Delegate(callbacks, verbose=log_events)

        self._get_optimizer: T.Final = optimizer
        self._get_scheduler: T.Final = scheduler
        self._flops = 0
        self._globalstep_last_logged = -1
        self._recover_path = None  # See: `recover` method

        self._xlr.register_for_checkpointing(self._state)

        self._edge(Event.ON_CREATE)
        self._mem_tracker.stop_and_update_metrics("init")

    status = StatusDescriptor(EngineStatus, default=EngineStatus(0))

    ##############
    # Public API #
    ##############

    @override
    def __str__(self) -> str:
        args = ", ".join(
            [
                f"{k}={v}"
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

    @functools.cached_property
    def root_path(self) -> file_io.Path:
        """
        Returns the local path to the root directory of this engine as a ``pathlib.Path`` class.
        """
        return file_io.Path(self._params.root).resolve()

    @functools.cached_property
    def logs_path(self) -> file_io.Path:
        """
        Returns the local path to the logs directory of this engine as a ``pathlib.Path`` class.
        """
        return file_io.Path(self._xlr.logging_dir)

    @functools.cached_property
    def outputs_path(self) -> file_io.Path:
        """
        Returns the local path to the outputs directory of this engine as a ``pathlib.Path`` class.
        """
        return file_io.Path(self._xlr.project_dir)

    def recover(self, model: nn.Module | None = None, checkpoint: str | file_io.Path | None = None) -> None:
        """
        Recover a model's state and the engine's state from the given checkpoint. The model is prepared in
        evaluation mode.
        """

        if checkpoint is not None:
            self._recover_path = str(checkpoint)

        if model is not None:
            self._xlr.prepare_model(model, evaluation_mode=True)
        self._load_state(self._recover_path)  # type: ignore

        _logger.info("Recovered engine state at step %d", self._state.step)

        return None

    @status(EngineStatus.IS_TRAINING_RUN)
    def train(
        self,
        model_factory: ModelFactory[Trial, nn.Module],
        *,
        trial: Trial | None = None,
        stage: int | str | None = None,
        weights: str | None = None,
    ) -> nn.Module:
        """
        Train a model.

        Parameters
        ----------
        model_factory : ModelFactory[Trial, Module]
            A factory function that returns a model.
        loader_factory : DataLoaderFactory[TensorDict]
            A factory function that returns a data loader.
        checkpoint : str | Literal["best"], optional
            A checkpoint to resume training from.
        trial : Trial, optional
            The trial to train.
        stage : int | str, optional
            The stage to train. If not specified, the current stage is used.
        weights : str, optional
            Path to a checkpoint to load **model** weights from.
        """

        # Memory metrics - must set up as early as possible
        self._mem_tracker.start("train")

        # Stage resolver
        if stage is None:
            stage = self._state.stage
        elif isinstance(stage, str):
            stage = self._stages.index(stage)

        # Divide batch size over the amount of processes
        assert self._params.train_batch_size % self._xlr.num_processes == 0, (
            f"Training batch size {self._params.train_batch_size} must be divisible over the amount of "
            f"processes {self._xlr.num_processes}."
        )
        loader_factory = self._loaders[self._stages[stage]]
        loader = loader_factory(self._params.train_batch_size // self._xlr.num_processes)

        # Infer the number of steps/updates per epoch
        steps_per_epoch = len(loader) // self._xlr.num_processes
        updates_per_epoch = math.ceil(steps_per_epoch / self._params.gradient_accumulation_steps)

        _logger.debug(
            (
                "Loader contains %d batches (%d processes, %d steps per epoch, %d accumulation steps, "
                f"%d steps per optimization)"
            ),
            len(loader),
            self._xlr.num_processes,
            steps_per_epoch,
            self._params.gradient_accumulation_steps,
            updates_per_epoch,
        )

        # Instantiate a model through the model factory
        model = model_factory(trial)
        scheduled_epochs = self._params.get_train_epochs(steps_per_epoch)
        optimizer = self._get_optimizer(model)
        scheduler, train_epochs = self._get_scheduler(optimizer, scheduled_epochs, updates_per_epoch)

        _logger.info(f"Training for {train_epochs} total epochs.")

        # Reset the state
        self._state.reset()
        self._state.stage = stage
        self._state.logging_steps = self._params.logging_steps
        self._state.eval_steps = self._params.get_eval_interval_steps(steps_per_epoch)
        self._state.save_steps = self._params.get_save_interval_steps(steps_per_epoch)
        self._state.train_steps = self._params.get_train_steps(steps_per_epoch)
        self._state.best_metric = None

        if trial is not None:
            self._state.trial_name = trial.name
            self._state.trial_params = trial.params
        else:
            self._state.trial_name = "training"
            self._state.trial_params = {}

        return self._train_loop(
            loader,
            model,
            optimizer,
            scheduler,
            trial=trial,
            weights=weights,
        )

    @status(EngineStatus.IS_EVALUATION_RUN)
    @torch.no_grad()
    def evaluate(
        self,
        model_factory: ModelFactory[Trial, nn.Module],
        *,
        trial: Trial | None = None,
        prefix: str = "evaluation",
        weights: str | None = None,
    ) -> dict[str, float]:
        _logger.info("*** Starting evaluation ***")

        metrics_overall = {}

        for loader_key, handlers in self._evaluators.items():
            _logger.info(f"Running inference on loader '%s' for %d handlers", loader_key, len(handlers))

            prefix_suite = "/".join([prefix, loader_key])

            torch.cuda.empty_cache()
            gc.collect()
            # Memory metrics - must set up as early as possible
            self._mem_tracker.start("eval")

            model = model_factory(trial)

            loader_factory = self._loaders[loader_key]
            loader = loader_factory(self._params.infer_batch_size)

            metrics, samples_processed = self._inference_loop(
                model, loader, prefix=prefix_suite, handlers=handlers, weights=weights
            )

            self._training_log(metrics)
            self._edge(Event.ON_EVALUATE, metrics=metrics)
            self._mem_tracker.stop_and_update_metrics("eval", metrics)

            del loader
            del model

            for metric_key in list(metrics.keys()):
                if not metric_key.startswith(prefix_suite):
                    metrics[prefix_suite] = metrics.pop(metric_key)

            metrics_overall.update(metrics)

        return metrics_overall

    @status.assert_status(~(EngineStatus.IS_TRAINING_RUN | EngineStatus.IS_EVALUATION_RUN))
    @status(EngineStatus.IS_PREDICTION_RUN)
    @torch.no_grad()
    def predict(self, model: nn.Module, datapipe: Dataset, *, prefix: str = "pred") -> TensorDict:
        raise NotImplementedError("TODO: Implement prediction")

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
        self._signal = self._delegate(event, self._params, self._state, self._signal, **kwargs)

    def _get_config_object(self) -> dict[str, T.Any]:
        """
        Attempt to locate the configuration YAML file for the current project.
        If that does not exist, return None. If it does exist, return the configuration object.
        """
        from unipercept.config import LazyConfig

        lazy_path = file_io.Path(self._xlr.project_dir) / ".." / "config.yaml"
        if not lazy_path.exists():
            return {}

        try:
            lazy = LazyConfig.load(str(lazy_path))
        except Exception as e:
            _logger.warn(f"Could not load configuration from {lazy_path}: {e}")
            return {}

        lazy_obj = OmegaConf.to_container(lazy, resolve=False)
        assert isinstance(lazy_obj, dict)
        return T.cast(dict[str, T.Any], lazy_obj)

    def _start_experiment_trackers(self, *, restart: bool = True) -> None:
        """
        Initialize the experiment trackers, e.g. WandB, TensorBoard. This should be called at the beginning of training
        and inference.

        Parameters
        ----------
        model : nn.Module
            The model to be watched by the loggers. Only applicable to some loggers (e.g. WandB)
        restart : bool, optional
            Whether to restart the loggers, by default True. Can be set to False to continue logging to the same
            trackers, e.g. when running an inference loop during training.
        """

        if EngineStatus.EXPERIMENT_TRACKERS_STARTED in self.status and not restart:
            _logger.debug("Trackers already started, skipping setup")
            return
        else:
            self.status |= EngineStatus.EXPERIMENT_TRACKERS_STARTED

        _logger.debug("Setting up experiment trackers")

        # Session name is organised in /group_part1/group_part2/.../session_name
        session_name_parts = self._params.session_name.split("/")
        if len(session_name_parts) > 1:
            session_group = "-".join(session_name_parts[:-1])
        else:
            session_group = "ungrouped"

        # Determine the job type from the status
        if self.status & EngineStatus.IS_TRAINING_RUN:
            job_type = "train"
        elif self.status & EngineStatus.IS_EVALUATION_RUN:
            job_type = "eval"
        elif self.status & EngineStatus.IS_PREDICTION_RUN:
            job_type = "pred"
        else:
            job_type = "misc"

        # WandB-specific setup
        def __wandb_sanitize(s):
            for c in R"/\#?%:":
                s = s.replace(c, "-")
            return s

        __wandb_kwargs = {
            "name": __wandb_sanitize(self._params.session_name.replace("/", " ")),
            "job_type": job_type,
            "group": session_group,
            "notes": self._params.notes,
            "tags": [f"stage_{self._state.stage}"] + list(self._params.tags),
            "id": __wandb_sanitize(f"{self._params.project_name}-{self._params.session_name}-{job_type}"),
            "save_code": False,
        }

        # Accelerate handles the experiment trackers for us
        self._xlr.init_trackers(
            self._params.project_name.replace("/", " "),
            config=self._get_config_object(),
            init_kwargs={"wandb": __wandb_kwargs},
        )
        self._edge(Event.ON_TRACKERS_SETUP)
        self._xlr.wait_for_everyone()

    def _train_loop(
        self,
        loader: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: TimmScheduler,
        *,
        weights: str | None = None,
        **kwargs,
    ) -> nn.Module:
        """
        The main training loop. This method is called by the `train` method.
        """

        self._xlr.free_memory()
        self._start_experiment_trackers()

        if self._params.convert_sync_batchnorm:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if weights is not None:
            model = self._load_weights(weights, model)
        model = self._xlr.prepare_model(model)
        loader, scheduler, optimizer = self._xlr.prepare(loader, scheduler, optimizer)

        # First load the initial weights, then the state
        self._load_state(None)  # type: ignore

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

        total_sesssion_samples = 0
        total_session_steps = 0

        steps_per_epoch = len(loader)
        start_epoch = math.floor(self._state.epoch)
        steps_trained_in_current_epoch = self._xlr.step  # int((self._state.epoch - start_epoch) * steps_per_epoch)

        assert steps_trained_in_current_epoch == int((self._state.epoch - start_epoch) * steps_per_epoch), (
            f"Expected {steps_trained_in_current_epoch} to be equal to "
            f"int(({self._state.epoch} - {start_epoch}) * {steps_per_epoch})"
        )

        train_epochs = math.ceil(self._state.train_steps / steps_per_epoch)

        # Check if the loader requires an epochs state
        if hasattr(loader, "epoch"):
            setattr(loader, "epoch", start_epoch)
        if hasattr(loader.sampler, "epoch"):
            setattr(loader.sampler, "epoch", start_epoch)

        # -- Epoch outer loop ------------------------------------------------------------------------------------------

        if self._xlr.is_main_process:
            _logger.info(f"Starting at epoch {start_epoch} at step {self._xlr.step}")

        for epoch in range(start_epoch, train_epochs):
            # Set the epoch iterator to the original dataloader
            epoch_iterator = loader

            self._edge(Event.ON_TRAIN_EPOCH_BEGIN)

            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                _logger.debug("Skipping the first %d steps in the current epoch", steps_trained_in_current_epoch)

                epoch_iterator = self._xlr.skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0

            steps_in_epoch = len(epoch_iterator)

            # -- Epoch inner loop --------------------------------------------------------------------------------------

            step = -1  # If the value of the iterator is still -1 after the loop, something went wrong
            for step, inputs in enumerate(epoch_iterator):
                assert isinstance(inputs, InputType), f"Expected InputType, got {type(inputs)}"

                total_sesssion_samples += 1

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % self._params.gradient_accumulation_steps == 0:
                    self._edge(Event.ON_TRAIN_STEP_BEGIN)

                with self._xlr.accumulate(model):
                    tr_loss_step = self._training_step(model, inputs)

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
                    tr_loss_step_value = tr_loss_step.get(k, torch.tensor(torch.nan, device=tr_loss_step.device))
                    if self._params.logging_nan_inf_filter and (
                        torch.isnan(tr_loss_step_value) or torch.isinf(tr_loss_step_value)
                    ):
                        tr_loss_value += tr_loss_value / (1 + self._state.step - self._globalstep_last_logged)  # type: ignore
                    else:
                        tr_loss_value += tr_loss_step_value

                # Compute flops
                self._flops += float(_flops(model, inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= self._params.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_sesssion_samples % self._params.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # Gradient clipping
                    if self._params.max_grad_norm is not None and self._params.max_grad_norm > 0:
                        if hasattr(optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            optimizer.clip_grad_norm(self._params.max_grad_norm)  # type: ignore
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self._params.max_grad_norm)  # type: ignore
                        else:
                            self._xlr.clip_grad_norm_(
                                model.parameters(),
                                self._params.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer.step()
                    if not self._xlr.optimizer_step_was_skipped:
                        scheduler.step_update(self._state.step, metric=None)  # TODO metric is not used
                    else:
                        _logger.debug("Step was skipped")
                    optimizer.zero_grad()
                    self._state.step += 1
                    self._state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch

                    del inputs

                    self._edge(Event.ON_TRAIN_STEP_END, model=model, optimizer=optimizer)
                    self._train_handle_log_save_eval(tr_loss, model, optimizer, **kwargs)
                else:
                    self._edge(Event.ON_TRAIN_SUBSTEP_END)

                total_session_steps += 1

                if self._signal.should_epoch_stop or self._signal.should_training_stop:
                    _logger.debug(f"Stopping epoch @ step {step} due to signal {self._signal}")
                    break

            # -- End of epoch ------------------------------------------------------------------------------------------
            if step < 0:
                assert tr_loss is None
                _logger.warning("Epoch was ended without running any steps.")
                self._signal.should_training_stop = True
                tr_loss = TensorDict.from_dict({})
            else:
                assert tr_loss is not None

            scheduler.step(round(self._state.epoch), metric=None)

            self._edge(Event.ON_TRAIN_EPOCH_END)
            self._train_handle_log_save_eval(tr_loss, model, optimizer, **kwargs)

            epochs_trained += 1

            if self._signal.should_training_stop:
                break

        # -- End of training -------------------------------------------------------------------------------------------

        _logger.info("\n\nTraining completed.\n\n")

        # Compute flops
        self._state.total_flops += self._flops
        self._flops = 0

        # Report final metrics
        metrics: dict[str, T.Any] = {}
        metrics.update(
            _build_speed_metrics(
                "training",
                time_start,
                sample_amount=total_sesssion_samples,
                step_amount=total_session_steps,
            )
        )
        metrics["total_flops"] = self._state.total_flops

        for k in list(metrics.keys()):
            if not k.startswith("engine/"):
                metrics["engine/" + k] = metrics.pop(k)

        self._mem_tracker.stop_and_update_metrics("train", metrics)
        self._training_log(metrics)
        self._edge(Event.ON_TRAIN_END)

        return model

    def _train_handle_log_save_eval(
        self,
        tr_loss: TensorDict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        trial: Trial | None,
    ) -> None:
        """
        Called at the end of every step and epoch to log, save, and evaluate the model.
        Steps could be skipped depending on the configuration.
        """

        if self._signal.should_log:
            assert (
                self._state.step != self._globalstep_last_logged
            ), "No new logs should be created when nothing changed"

            logs: dict[str, float] = {}
            logs["optimizer/lr"] = _get_learning_rate(optimizer)

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = {k: self._xlr.gather(l).mean().item() for k, l in tr_loss.items()}

            # reset tr_loss to zero
            # tr_loss -= tr_loss
            tr_loss.apply_(lambda _l: _l - _l)

            for k, v in tr_loss_scalar.items():
                logs["losses/" + k] = round(v / (self._state.step - self._globalstep_last_logged), 4)

            self._globalstep_last_logged = self._state.step
            # self.store_flops()
            self._training_log(logs)

        if self._signal.should_save:
            _logger.info("Saving state and model at step %d (epoch %d)", self._state.step, self._state.epoch)

            # Save the training state (for recovery on in this environment)
            try:
                state_path = self._save_state(None)
            except Exception as e:
                state_path = None
                _logger.error("Failed to save accelerator state: %s", e, stacklevel=2)

            # Save the (unwrapped) model
            try:
                model_path = self._save_weights(None, model)
            except Exception as e:
                model_path = None
                _logger.error("Failed to save model: %s", e, stacklevel=2)

            # Report event
            self._edge(Event.ON_SAVE, model_path=model_path, state_path=state_path)

        if self._signal.should_evaluate:
            _logger.info("Starting evaluation cycle @ step %d / epoch %d", self._state.step, self._state.epoch)

            self.evaluate(lambda _: model, trial=trial)

    def _training_log(self, logs: dict[str, float | str | int | bool]) -> None:
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
        logs["engine/epoch_step"] = self._xlr.step
        logs["engine/status"] = self.status

        self._edge(Event.ON_LOG, logs=logs)  # NOTE: logs may be updated in-place

        self._state.log_history.append(logs)
        if len(self._state.log_history) > self._params.logging_history:
            self._state.log_history.pop(0)
        self._xlr.log(logs, step=self._state.step)

    def _training_step(self, model: nn.Module, inputs: InputType) -> TensorDict:
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
            self._xlr.backward(loss_tensor.sum())
        else:
            self._xlr.backward(loss_tensor, gradient=torch.ones_like(loss_tensor))

        losses.detach_()
        return losses.apply(lambda _l: _l / self._params.gradient_accumulation_steps)

    def _load_weights(self, path: str, model: nn.Module) -> nn.Module:
        """
        Load the model checkpoint at the given path.

        Parameters
        ----------
        path : str
            The path to the model checkpoint. Skipped when `None`.
        model : nn.Module
            The model to load the checkpoint into.

        Returns
        -------
        nn.Module
            The model with the loaded checkpoint.
        """
        from accelerate import load_checkpoint_and_dispatch

        _logger.debug("Loading weights from %s", path)
        return load_checkpoint_and_dispatch(model, file_io.get_local_path(path))

    def _save_weights(self, path: T.Optional[str], model: nn.Module) -> str:
        """
        Save a model, unwrapping it from the accelerator

        Parameters
        ----------
        output_dir
            The directory to save the model checkpoints to.
        """
        if path is None:
            path = os.path.join(self._xlr.project_dir, "models", "step-" + str(self._state.step))

        if check_main_process():
            os.makedirs(path, exist_ok=True)
            model = self._xlr.unwrap_model(model)
            self._xlr.save_model(model, save_directory=path, safe_serialization=True)

        return path

    def _load_state(self, path: T.Optional[str]) -> None:
        """
        Load the engine state from the given path, if no path is given, the last checkpoint is used.
        """
        if path is not None:
            _logger.debug("Loading engine state at %s", path)
            path = file_io.get_local_path(path)
        elif self._recover_path is not None:
            _logger.debug("Loading engine state from recovery path: %s", path)
            path = file_io.get_local_path(self._recover_path)
        else:
            auto_dir = os.path.join(self._xlr.project_dir, "checkpoints")
            if file_io.isdir(auto_dir) and len(file_io.ls(auto_dir)) > 0:
                _logger.debug("No engine state path given and no automatic checkpoints found.")
                return
            else:
                _logger.debug("No engine state path given. Defaulting to last automatic checkpoint in %s", auto_dir)
        self._xlr.load_state(path)  # type: ignore

    def _save_state(self, path: T.Optional[str]) -> str:
        """
        Save the engine state for recovery/resume. Sometimes called a 'checkpoint'.
        """
        if path is not None:
            path = file_io.get_local_path(path)
        return self._xlr.save_state(path)  # type: ignore

    def _inference_gather_total_batches(self, dataloader: torch.utils.data.DataLoader) -> tuple[list[int], int]:
        a = len(dataloader)

        # Gather the size of dataloaders across all processes
        a_dist = torch.tensor([a], dtype=torch.int64, device=self._xlr.device)
        a_dist = self._xlr.gather(a_dist)
        assert isinstance(a_dist, torch.Tensor), f"Expected Tensor, got {type(a_dist)}"

        # Compute the offsets list
        a_off: list[int] = a_dist.cumsum(0).tolist()
        a_off = [0] + a_off[:-1]
        assert len(a_off) == self._xlr.num_processes, f"Expected {self._xlr.num_processes} offsets, got {len(a_off)}"

        # Compute total amount of samples
        a_total = int(a_dist.sum().item())

        _logger.debug("Gathered sample offsets: %s (%d total samples)", a_off, a_total)

        return a_off, a_total

    @torch.inference_mode()
    def _inference_loop(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        prefix: str,
        handlers: T.Sequence[Evaluator] | None = None,
        results_path: T.Optional[file_io.PathLike] = None,
        *,
        weights: T.Optional[file_io.PathLike] = None,
    ) -> tuple[dict[str, T.Any], int]:
        """
        Evaluation loop, which roughly follows the folowing procedure:

            (1) prepare the model and data loader
            (2) run prediction on the dataset and feed results through each evaluator's preprocessing function
            (3) iterate the dataset again, and run the evaluation function of each evaluator
        """

        self._start_experiment_trackers(restart=False)

        _logger.info("***** Running inference *****")

        torch.cuda.empty_cache()
        gc.collect()

        # Find the size of the dataset *before* preparation to correctly map each output
        _, batch_total = self._inference_gather_total_batches(dataloader)
        samples_total = batch_total * self._params.infer_batch_size

        _logger.debug(f"Expecting {samples_total} samples in total across {batch_total} batches")

        # Prepare data loader
        dataloader = self._xlr.prepare_data_loader(dataloader)

        # Prepare model
        if weights is not None:
            model = self._load_weights(weights, model)
        if self._xlr.unwrap_model(model) is model:
            _logger.info(f"Preparing model for evaluation: {model.__class__.__name__}")
            model = self._xlr.prepare_model(model, evaluation_mode=True)
        else:
            _logger.info(f"Model is already prepared for evaluation: {model.__class__.__name__}")

        # If not in training status, apply the proper datatype
        if not (self.status & EngineStatus.IS_TRAINING_RUN):
            match self._params.inference_precision:
                case InferencePrecision.FULL_FP16:
                    model = model.to(dtype=torch.float16, device=self._xlr.device)
                case InferencePrecision.FULL_BF16:
                    model = model.to(dtype=torch.bfloat16, device=self._xlr.device)
                case InferencePrecision.DEFAULT:
                    pass
                case _:
                    raise ValueError(f"Invalid inference precision: {self._params.inference_precision}")

        model.eval()

        # Global start time
        t_start_all = time.time()

        # Output memory
        if results_path is None:
            results_remove_on_exit = True
            results_path = file_io.Path(
                f"//scratch/{self._params.project_name}/{self._params.session_name}/{prefix}-results.h5"
            )
        else:
            results_remove_on_exit = False
        results_mem = PersistentTensordictWriter(str(results_path), samples_total)

        try:
            # Prediction
            _logger.info(f"Running inference loop on {self._xlr.num_processes} processes.")

            samples_processed = 0
            batch_size = self._params.infer_batch_size
            # write_index = batch_offsets[self._xlr.process_index] * batch_size

            self._edge(Event.ON_INFERENCE_BEGIN, loader=dataloader)
            timings = ProfileAccumulator()

            for inputs in dataloader:
                samples_in_batch = inputs.batch_size[0]
                assert samples_in_batch <= batch_size, f"Expected batch size {batch_size}, got {samples_in_batch}"

                # Prediction step - i.e. run the model in inference mode
                with profile(timings, "model"):
                    outputs = self._inference_step(model, inputs)

                # Prepare for evaluation
                with profile(timings, "update"):
                    KEY_VALID = "valid"
                    results_merged = TensorDict(
                        {KEY_VALID: torch.ones(samples_in_batch, dtype=torch.bool, device=self._xlr.device)},
                        [samples_in_batch],
                        self._xlr.device,
                        names=["B"],
                    )
                    for evaluator in handlers:
                        evaluator.update(results_merged, inputs=inputs, outputs=outputs)

                # Gather results
                with profile(timings, "write"):
                    results_mem.add(results_merged)

                    # write_index += samples_in_batch
                    samples_processed += samples_in_batch
                self._edge(Event.ON_INFERENCE_STEP, loader=dataloader, inputs=inputs, outputs=outputs)

            results_mem.write()

            _logger.info(
                f"Finished inference, profiling report on process %d/%d:\n%s",
                self._xlr.process_index + 1,
                self._xlr.num_processes,
                timings.to_summary().to_markdown(index=True, floatfmt=".3f"),
            )

            self._edge(Event.ON_INFERENCE_END, timings=timings, results=results_mem, results_path=results_path)

            # Wait for everyone to finish writing
            self._xlr.wait_for_everyone()

            # Run metric computations
            metrics: dict[str, T.Any] = {}
            visuals: dict[str, pil_image.Image] = {}
            if self._xlr.is_main_process:
                metrics.update(
                    _build_speed_metrics(
                        prefix,
                        t_start_all,
                        sample_amount=samples_processed,
                        step_amount=math.ceil(samples_processed * self._xlr.num_processes),
                    )
                )
                assert results_mem is not None, "Expected results memory to be initialized"
                for evaluator in handlers or []:
                    _logger.debug(f"Running evaluation handler: {evaluator}")
                    metrics.update(evaluator.compute(results_mem.tensordict, device=self._xlr.device))
                    visuals.update(evaluator.plot(results_mem.tensordict))

                # Store visualizations
                self._store_visualizations(visuals, prefix=prefix)
        finally:
            # Remove the memmap file
            results_mem.close()
            self._xlr.wait_for_everyone()
            if self._xlr.is_main_process and results_remove_on_exit:
                assert results_path is not None
                _logger.debug(f"Cleaning up results file: {results_path}")
                shutil.rmtree(results_path, ignore_errors=True)

        # Store metrics
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{prefix}/jit_compilation_time"] = self.jit_compilation_time  # type: ignore

        # Prefix all keys
        _enforce_prefix(metrics, prefix)

        return metrics, samples_processed

    def _store_visualizations(self, visuals: dict[str, pil_image.Image], prefix: str) -> None:
        """
        Store visualizations that are provided as a mapping of (key) -> (PIL image).
        """

        _logger.info(f"Storing visualizations ({len(visuals)} total)")

        for key, img in visuals.items():
            if self._params.eval_write_visuals:
                img_path = (
                    file_io.Path(self._xlr.project_dir) / "visuals" / f"{prefix}-{self._state.step}" / f"{key}.png"
                )
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(img_path)

            wandb_run = self._xlr.get_tracker("wandb")
            if wandb_run is not None:
                wandb_run.log({f"{prefix}/{key}": wandb.Image(img)}, step=self._state.step)

    def _inference_step(
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

    def _default_setup(self):
        """
        Sets default configuration values in third-party libraries.
        """

        # See: https://pytorch.org/docs/stable/notes/multiprocessing.html
        slurm = unipercept.integrations.slurm_integration.SLURMEnvironment()

        if slurm.is_slurm_job:
            N_M = slurm.cpus_per_gpu
        else:
            N = multiprocessing.cpu_count()
            M = self._xlr.num_processes
            N_M = math.floor(N / M)
        torch.set_num_threads(N_M)


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
    prefix: str, start_time: float, *, sample_amount: int | None = None, step_amount: int | None = None
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


def _enforce_prefix(metrics: dict[str, T.Any], prefix: str, sep: str = "/") -> None:
    """
    Enforce a prefix on all keys in `metrics`.
    """
    if not prefix.endswith(sep):
        prefix = prefix + sep
    for key in list(metrics.keys()):
        if key.startswith(prefix):
            continue
        metrics[prefix + key] = metrics.pop(key)


def _build_accelerator(params: EngineParams) -> accelerate.Accelerator:
    """
    Builds the Accelerator object.

    Parameters
    ----------
    config : EngineConfig
        The configuration object.

    Returns
    -------
    accelerate.Accelerator
        The accelerator object.
    """
    from accelerate.accelerator import ProjectConfiguration
    from accelerate.utils import DistributedDataParallelKwargs

    root = file_io.Path(params.root).resolve()

    project_dir = root / "outputs"
    logging_dir = root / "logs"

    project_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    acc = accelerate.Accelerator(
        project_dir=project_dir,
        project_config=ProjectConfiguration(
            project_dir=str(project_dir), logging_dir=str(logging_dir), automatic_checkpoint_naming=True, total_limit=4
        ),
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=params.find_unused_parameters,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        ],
        step_scheduler_with_optimizer=False,
        log_with=list(params.trackers),
        dispatch_batches=False,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
    )
    return acc
