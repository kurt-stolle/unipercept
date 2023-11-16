"""
The `Trainer` class is the main class to handle training and evaluation of any kind of models with any kind of datasets.
"""

from __future__ import annotations

import dataclasses as D
import enum as E
import functools
import math
import operator
import os
import shutil
import time
import typing as T
import warnings

import accelerate
import accelerate.utils
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import wandb
from matplotlib.pyplot import step
from PIL import Image as pil_image
from tensordict import MemmapTensor, TensorDict, TensorDictBase
from timm.scheduler.scheduler import Scheduler as TimmScheduler
from torch.utils.data import Dataset
from typing_extensions import override
from unicore import file_io
from unicore.utils.status import StatusDescriptor
from unicore.utils.tensorclass import Tensorclass

from ..utils.logutils import get_logger
from ..utils.seed import set_seed
from ._optimizer import OptimizerFactory
from ._scheduler import SchedulerFactory
from ._trial import SearchBackend, Trial
from ._types import DataLoaderFactory, ModelFactory
from .callbacks import CallbackType, Delegate, Event, Signal, TrainState
from .config import InferencePrecision, TrainConfig
from .debug import DebugMode, DebugUnderflowOverflow
from .memory import MemoryTracker

if T.TYPE_CHECKING:
    from ..evaluators import Evaluator
    from ..model import InputData, ModelOutput

MaybeTensorType = T.TypeVar("MaybeTensorType", bound=torch.Tensor | None)

__all__ = ["Trainer", "TrainerStatus"]
_logger = get_logger(__name__)


def _build_accelerator(config: TrainConfig) -> accelerate.Accelerator:
    from accelerate import PartialState
    from accelerate.accelerator import ProjectConfiguration
    from accelerate.utils import DistributedDataParallelKwargs

    root = file_io.Path(config.root).resolve()

    project_dir = root / "outputs"
    logging_dir = root / "logs"

    if PartialState().is_main_process:
        project_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Using directory: %s", str(root))

    acc = accelerate.Accelerator(
        project_dir=project_dir,
        project_config=ProjectConfiguration(
            project_dir=str(project_dir), logging_dir=str(logging_dir), automatic_checkpoint_naming=True, total_limit=4
        ),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=config.find_unused_parameters)],
        step_scheduler_with_optimizer=False,
        log_with=config.trackers,
        dispatch_batches=False,
    )

    from pprint import pformat

    _logger.info("Using accelerator: \n%s", pformat(D.asdict(acc.project_configuration)))
    _logger.info("Current process: %d / %d", acc.process_index, acc.num_processes)

    return acc


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
    result = {f"{prefix}_time": round(delta_time, 4)}
    if delta_time == 0:
        return result
    if sample_amount is not None:
        samples_per_second = sample_amount / delta_time
        result[f"{prefix}_samples_per_second"] = round(samples_per_second, 3)
    if step_amount is not None:
        steps_per_second = step_amount / delta_time
        result[f"{prefix}_steps_per_second"] = round(steps_per_second, 3)
    return result


InputType: T.TypeAlias = TensorDict | Tensorclass
TrainingOutputType: T.TypeAlias = T.Dict[str, torch.Tensor]
InferenceOutputType: T.TypeAlias = T.Sequence[T.Dict[str, T.Any]]


class TrainerStatus(E.IntFlag):
    """
    The current status of the trainer. This status is not part of the persistent/stored state. Used internally for
    control flow, e.g. in cases where the evaluation loop is ran while also training.
    """

    TRAINING = E.auto()
    EVALUATION = E.auto()
    INFERENCE = E.auto()
    TUNING = E.auto()


class Trainer:
    """
    Training loop implementation.
    """

    def __init__(
        self,
        *,
        config: TrainConfig,
        optimizer: OptimizerFactory,
        scheduler: SchedulerFactory,
        callbacks: T.Sequence[CallbackType | type[CallbackType]],
        evaluators: T.Sequence[Evaluator] | None = None,
        log_events: bool = False,
    ):
        self._mem_tracker = MemoryTracker(False)
        self._mem_tracker.start("init")  # must set up as early as possible

        _logger.info("Initializing Trainer: %s @ %s", config.project_name, config.session_name)

        self._config: T.Final[TrainConfig] = config
        self._xlr = _build_accelerator(config)
        self._state = TrainState()
        self._evaluators = evaluators or []

        self._seed()

        self._signal = Signal()
        self._delegate = Delegate(callbacks, verbose=log_events)

        self._get_optimizer: T.Final = optimizer
        self._get_scheduler: T.Final = scheduler
        self._flops = 0
        self._globalstep_last_logged = -1
        self._past = None
        self._recover_path = None  # See: `recover` method

        self._xlr.register_for_checkpointing(self._state)

        self._event(Event.ON_INIT_END)
        self._mem_tracker.stop_and_update_metrics("init")

    status = StatusDescriptor[TrainerStatus](default=TrainerStatus(0))

    ##############
    # Public API #
    ##############

    @override
    def __str__(self) -> str:
        args = ", ".join(
            [
                f"{k}={v}"
                for k, v in {
                    "config": str(self._config),
                    "state": str(self._state),
                    "status": str(self.status),
                }.items()
            ]
        )

        return f"{self.__class__.__name__}(\n{args}\n)"

    @override
    def __repr__(self) -> str:
        return str(self)

    @functools.cached_property
    def path(self) -> file_io.Path:
        """
        Returns the local path to the root directory of this trainer as a ``pathlib.Path`` class.
        """
        return file_io.Path(self._config.root).resolve()

    def recover(self, model: nn.Module = None, checkpoint: str | file_io.Path | None = None) -> None:
        """
        Recover a model's state and the trainer's state from the given checkpoint. The model is prepared in
        evaluation mode.
        """

        if checkpoint is not None:
            self._recover_path = str(checkpoint)

        if model is not None:
            self._xlr.prepare_model(model, evaluation_mode=True)
        self._xlr.load_state(self._recover_path)  # type: ignore

        _logger.info("Recovered trainer state at step %d", self._state.step)

        return None

    @status(TrainerStatus.TRAINING)
    def train(
        self,
        model_factory: ModelFactory[Trial, nn.Module],
        loader_factory: DataLoaderFactory[TensorDict],
        *,
        checkpoint: str | T.Literal["best"] | None = None,
        trial: Trial | None = None,
        evaluation_loader_factory=None,
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
        evaluation_handlers
            A list of evaluators to use for evaluation.
        evaluation_loader_factory : DataLoaderFactory[TensorDict], optional
            A factory function that returns a data loader for evaluation.
        """

        if checkpoint is None:
            checkpoint = self._recover_path

        self.status |= TrainerStatus.TRAINING

        # Memory metrics - must set up as early as possible
        self._mem_tracker.start("train")

        # Divide batch size over the amount of processes
        assert self._config.train_batch_size % self._xlr.num_processes == 0, (
            f"Training batch size {self._config.train_batch_size} must be divisible over the amount of "
            f"processes {self._xlr.num_processes}."
        )
        loader = loader_factory(self._config.train_batch_size // self._xlr.num_processes)

        # Infer the number of steps/updates per epoch
        steps_per_epoch = len(loader) // self._xlr.num_processes
        updates_per_epoch = math.ceil(steps_per_epoch / self._config.gradient_accumulation_steps)

        _logger.debug(
            (
                "Loader contains %d batches (%d processes, %d steps per epoch, %d accumulation steps, "
                f"%d steps per optimization)"
            ),
            len(loader),
            self._xlr.num_processes,
            steps_per_epoch,
            self._config.gradient_accumulation_steps,
            updates_per_epoch,
        )

        # Instantiate a model through the model factory
        model = model_factory(trial)
        scheduled_epochs = self._config.get_train_epochs(steps_per_epoch)
        optimizer = self._get_optimizer(model)
        scheduler, train_epochs = self._get_scheduler(optimizer, scheduled_epochs, updates_per_epoch)

        _logger.info(f"Training for {train_epochs} total.")

        # Reset the state
        self._state.reset()
        self._state.logging_steps = self._config.logging_steps
        self._state.eval_steps = self._config.get_eval_interval_steps(steps_per_epoch)
        self._state.save_steps = self._config.get_save_interval_steps(steps_per_epoch)
        self._state.train_steps = self._config.get_train_steps(steps_per_epoch)
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
            checkpoint=checkpoint,
            evaluation_loader_factory=evaluation_loader_factory,
        )

    @status(TrainerStatus.EVALUATION)
    def evaluate(
        self,
        model_factory: ModelFactory[Trial, nn.Module],
        loader_factory: DataLoaderFactory[InputType],
        *,
        trial: Trial | None = None,
        prefix: str = "eval",
    ) -> dict[str, float]:
        _logger.info("Starting evaluation")

        # Memory metrics - must set up as early as possible
        self._mem_tracker.start("eval")

        model = model_factory(trial)
        loader = loader_factory(self._config.infer_batch_size)
        metrics, samples_processed = self._inference_loop(
            model,
            loader,
            prefix=prefix,
            handlers=self._evaluators,
        )

        self._training_log(metrics)
        self._event(Event.ON_EVALUATE, metrics=metrics)
        self._mem_tracker.stop_and_update_metrics("eval", metrics)

        del loader
        del model

        return metrics

    @status.assert_status(~(TrainerStatus.TRAINING | TrainerStatus.EVALUATION))
    @status(TrainerStatus.INFERENCE)
    def predict(self, model: nn.Module, datapipe: Dataset, *, prefix: str = "pred") -> TensorDict:
        # Memory metrics - must set up as early as possible
        self._mem_tracker.start("pred")

        loader = torch.utils.data.DataLoader(datapipe)
        output = self._inference_loop(model, loader)

        self._event(Event.ON_PREDICT, metrics=output.metrics)
        self._mem_tracker.stop_and_update_metrics("pred", output.metrics)

        return output

    ############
    # Privates #
    ############

    def _seed(self) -> None:
        """
        Seed the random number generators.
        """
        set_seed(self._config.seed, fully_deterministic=self._config.full_determinism)

    def _event(self, event: Event, **kwargs) -> None:
        """
        Called internally on every event.
        """
        self._signal = self._delegate(event, self._config, self._state, self._signal, **kwargs)

    def _find_project_config(self) -> T.Optional[dict[str, int | str | bool | float]]:
        """
        Attempt to locate the configuration YAML file for the current project.
        If that does not exist, return None. If it does exist, return the parsed and flattened YAML.
        """
        from omegaconf import DictConfig

        from unipercept.utils.config import LazyConfig, flatten_config

        path = file_io.Path(self._xlr.project_dir) / ".." / "config.yaml"
        if not path.exists():
            return None

        try:
            config = LazyConfig.load(str(path))
        except Exception as e:
            _logger.warn(f"Could not load configuration from {path}: {e}")
            return None

        assert isinstance(config, DictConfig), f"Expected DictConfig, got {type(config)}"

        return flatten_config(config)

    def _init_trackers(self, model: nn.Module) -> None:
        """
        (Re)-Initialize the loggers, e.g. WandB, TensorBoard. This is called at the beginning of every training run.
        """
        from accelerate.tracking import WandBTracker

        # n = self._state.trial_name
        # assert n is not None, "Trial name is not initialized!"
        # TODO: load configuration from YAML
        self._xlr.init_trackers(
            self._config.project_name.replace("/", " "),
            config=self._find_project_config(),
            init_kwargs={
                "wandb": {
                    # "project": self._config.project_name,
                    "name": self._config.session_name,
                    # "job_type": "train",
                }
            },
        )

        # WandB specific configuration
        wandb_tracker: WandBTracker = self._xlr.get_tracker("wandb")  # type: ignore
        if wandb_tracker is not None:
            _logger.info("Using Weights and Biases tracker.")

            # wandb_tracker.store_init_configuration(self._state.trial_params)

            if self._xlr.is_main_process:
                wandb_tracker.tracker.watch(model, log="all", log_freq=self._config.logging_steps)
        else:
            _logger.info("Not using Weights and Biases tracker.")

        self._xlr.wait_for_everyone()

    def _train_loop(
        self,
        loader: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: TimmScheduler,
        *,
        checkpoint: str | None = None,
        **kwargs,
    ) -> nn.Module:
        """
        The main training loop, roughly following the implementation of the HuggingFace Trainer. The cycle is rougly as
        follows: (1) load the checkpoint, (2) train until an evaluation signal is given, (3) evaluate, (4) repeat or
        stop, (5) save the final checkpoint.
        """

        # Register and prepare using Accelerate
        self._xlr.free_memory()
        model = self._xlr.prepare_model(model)
        loader, scheduler, optimizer = self._xlr.prepare(loader, scheduler, optimizer)

        checkpoint_dir = os.path.join(self._xlr.project_dir, "checkpoints")
        if checkpoint is not None:
            self._xlr.load_state(checkpoint)  # type: ignore
        elif file_io.isdir(checkpoint_dir) and len(file_io.ls(checkpoint_dir)) > 0:
            self._xlr.load_state(None)  # type: ignore

        # Initialize loggers
        self._init_trackers(model)

        # Debugging
        # debug_overflow = DebugUnderflowOverflow(model)  # noqa
        if DebugMode.UNDERFLOW_OVERFLOW & self._config.debug != 0:
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

        self._event(Event.ON_TRAIN_BEGIN)

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

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self._config.past_index >= 0:
                self._past = None

            self._event(Event.ON_TRAIN_EPOCH_BEGIN)

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

                if step % self._config.gradient_accumulation_steps == 0:
                    self._event(Event.ON_TRAIN_STEP_BEGIN)

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
                    tr_loss_step_value = tr_loss_step.get(k)
                    if self._config.logging_nan_inf_filter and (
                        torch.isnan(tr_loss_step_value) or torch.isinf(tr_loss_step_value)
                    ):
                        tr_loss_value += tr_loss_value / (1 + self._state.step - self._globalstep_last_logged)  # type: ignore
                    else:
                        tr_loss_value += tr_loss_step_value

                # Compute flops
                self._flops += float(floating_point_ops(model, inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= self._config.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_sesssion_samples % self._config.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # Gradient clipping
                    if self._config.max_grad_norm is not None and self._config.max_grad_norm > 0:
                        if hasattr(optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            optimizer.clip_grad_norm(self._config.max_grad_norm)  # type: ignore
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self._config.max_grad_norm)  # type: ignore
                        else:
                            self._xlr.clip_grad_norm_(
                                model.parameters(),
                                self._config.max_grad_norm,
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

                    self._event(Event.ON_TRAIN_STEP_END)
                    self._maybe_log_save_evaluate(tr_loss, model, optimizer, **kwargs)
                else:
                    self._event(Event.ON_TRAIN_SUBSTEP_END)

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

            self._event(Event.ON_TRAIN_EPOCH_END)
            self._maybe_log_save_evaluate(tr_loss, model, optimizer, **kwargs)

            epochs_trained += 1

            if self._signal.should_training_stop:
                break

        # -- End of training -------------------------------------------------------------------------------------------

        if self._config.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

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

        self.is_in_train = False

        self._mem_tracker.stop_and_update_metrics("train", metrics)
        self._training_log(metrics)
        self._event(Event.ON_TRAIN_END)

        return model

    def _maybe_log_save_evaluate(
        self,
        tr_loss: TensorDict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        trial: Trial | None,
        evaluation_loader_factory: DataLoaderFactory | None,
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
            logs["learning_rate"] = get_learning_rate(optimizer)

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = {k: self._nested_gather(l).mean().item() for k, l in tr_loss.items()}

            # reset tr_loss to zero
            # tr_loss -= tr_loss
            tr_loss.apply_(lambda _l: _l - _l)

            for k, v in tr_loss_scalar.items():
                logs["loss/" + k] = round(v / (self._state.step - self._globalstep_last_logged), 4)

            self._globalstep_last_logged = self._state.step
            # self.store_flops()
            self._training_log(logs)

        if self._signal.should_evaluate:
            if evaluation_loader_factory is not None:
                _logger.info("Starting evaluation cycle @ step %d / epoch %d", self._state.step, self._state.epoch)

                self.evaluate(lambda _: model, evaluation_loader_factory, trial=trial)
            else:
                _logger.warning("Evaluation is requested but no evaluation data loader was provided.")

        if self._signal.should_save:
            _logger.info("Saving accelerator state @ step %d / epoch %d", self._state.step, self._state.epoch)
            try:
                self._xlr.save_state()
            except Exception as e:
                _logger.error("Failed to save accelerator state: %s", e, stacklevel=2)
            try:
                self._save_model(model, os.path.join(self._xlr.project_dir, "models", "step-" + str(self._state.step)))
            except Exception as e:
                _logger.error("Failed to save model: %s", e, stacklevel=2)
            self._event(Event.ON_SAVE)

    def _training_log(self, logs: dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Parameters
        ----------
        logs : dict[str, float]
            The logs to be logged.
        """
        logs["epoch"] = round(self._state.epoch, 2)
        logs["step"] = self._state.step
        logs["step_in_epoch"] = self._xlr.step
        logs["status"] = self.status

        self._event(Event.ON_LOG, logs=logs)  # NOTE: logs may be updated in-place

        self._state.log_history.append(logs)
        if len(self._state.log_history) > self._config.logging_history:
            self._state.log_history.pop(0)
        self._xlr.log(logs, step=self._state.step)

    def _training_step(self, model: nn.Module, inputs: InputType) -> TensorDict:
        """
        A single training step (forward + backward + update).
        """
        model.train()
        outputs: ModelOutput = model(inputs)

        loss_tensor = torch.stack([loss for loss in outputs.losses.values()])  # type: ignore

        if self._config.train_sum_losses:
            self._xlr.backward(loss_tensor.sum())
        else:
            self._xlr.backward(loss_tensor, gradient=torch.ones_like(loss_tensor))

        outputs.detach_()
        return outputs.losses.apply(lambda _l: _l / self._config.gradient_accumulation_steps)

    def _save_model(self, model: nn.Module, output_dir: T.Optional[str] = None):
        """
        Save a model checkpoint, unwrapping it from the accelerator

        Parameters
        ----------
        output_dir
            The directory to save the model checkpoints to.
        """
        if not self._xlr.is_main_process:
            return

        model = self._xlr.unwrap_model(model)
        if output_dir is None:
            output_dir = os.path.join(self._xlr.project_dir, "models", "step-" + str(self._state.step))
        os.makedirs(output_dir, exist_ok=True)

        self._xlr.save_model(model, output_dir, safe_serialization=True)

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

    def _inference_results_allocate(self, sample_amount: int, like: TensorDict) -> tuple[file_io.Path, TensorDict]:
        """
        Allocate a memmapped TensorDict to store results for evaluation.

        Parameters
        ----------
        sample_amount : int
            The total amount of samples that will be stored
        like : TensorDict
            A TensorDict with the same structure as the results TensorDict

        Returns
        -------
        TensorDict
            A TensorDict with the same structure as the results TensorDict, but with the first dimension set to the
            total amount of samples. Memory mapped.
        """
        _logger.debug(f"Allocating memory for {sample_amount} samples using template of batch size {like.batch_size}")

        prefix = (
            file_io.Path("//scratch/")
            / self._config.project_name
            / self._config.session_name
            / "evaluation"
            / f"step-{self._state.step}"
        )

        with self._xlr.main_process_first():
            if self._xlr.is_main_process:
                if prefix.exists():
                    _logger.debug("Clearing existing results directory: %s", prefix)
                    shutil.rmtree(prefix)
                else:
                    _logger.debug("Creating results directory: %s", prefix)
                prefix.mkdir(parents=True, exist_ok=False)

                mem = TensorDict(
                    {
                        k: MemmapTensor(
                            sample_amount,
                            *v.shape[1:],
                            dtype=v.dtype,
                            device="cpu",
                            filename=str(prefix / f"{k}.memmap"),
                        )
                        for k, v in like.items()
                    },
                    batch_size=[sample_amount],
                )
                mem.memmap_(prefix=str(prefix))
            else:
                _logger.debug("Using existing results directory on worker %d: %s", self._xlr.process_index, prefix)
                mem = TensorDict.load_memmap(prefix=str(prefix))
        # assert mem.is_memmap(), "Expected results memory to be memmapped"

        self._xlr.wait_for_everyone()
        return prefix, mem

    def _inference_loop(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        prefix: str = "eval",
        handlers: T.Sequence[Evaluator] = [],
    ) -> tuple[dict[str, T.Any], int]:
        """
        Evaluation loop, which roughly follows the folowing procedure:

            (1) prepare the model and data loader
            (2) run prediction on the dataset and feed results through each evaluator's preprocessing function
            (3) iterate the dataset again, and run the evaluation function of each evaluator
        """

        _logger.info("***** Running inference *****")

        if len(handlers) == 0:
            warnings.warn(
                (
                    "No evaluators were provided to the evaluation loop. "
                    "This is intended when only aiming to measure speed or memory usage."
                ),
                stacklevel=2,
            )

        # Find the size of the dataset *before* preparation to correctly map each output
        batch_offsets, batch_total = self._inference_gather_total_batches(dataloader)
        samples_total = batch_total * self._config.infer_batch_size

        _logger.debug(f"Expecting {samples_total} samples in total across {batch_total} batches")

        # Prepare model
        dataloader = self._xlr.prepare_data_loader(dataloader)
        if self._xlr.unwrap_model(model) is model:
            model = self._xlr.prepare_model(model, evaluation_mode=True)

        # If not in training status, apply the proper datatype
        if not (self.status & TrainerStatus.TRAINING):
            match self._config.inference_precision:
                case InferencePrecision.FULL_FP16:
                    model = model.to(dtype=torch.float16, device=self._xlr.device)
                case InferencePrecision.FULL_BF16:
                    model = model.to(dtype=torch.bfloat16, device=self._xlr.device)
                case InferencePrecision.DEFAULT:
                    pass
                case _:
                    raise ValueError(f"Invalid inference precision: {self._config.inference_precision}")

        model.eval()

        # Global start time
        t_start_all = time.time()

        # Reset past
        if self._config.past_index >= 0:
            self._past = None

        # Output memory
        results_mem = None
        results_prefix: str | None = None
        results_merged = None
        outputs = None

        # Prediction
        _logger.info(f"Running inference loop on {self._xlr.num_processes} processes.")

        samples_processed = 0
        batch_size = self._config.infer_batch_size
        write_index = batch_offsets[self._xlr.process_index] * batch_size
        for inputs in dataloader:
            samples_in_batch = inputs.batch_size[0]
            assert samples_in_batch <= batch_size, f"Expected batch size {batch_size}, got {samples_in_batch}"

            # Prediction step - i.e. run the model in inference mode
            outputs = self._inference_step(model, inputs)

            # Prepare for evaluation
            results_merged = TensorDict(
                {"valid": torch.ones(samples_in_batch, dtype=torch.bool)}, batch_size=inputs.batch_size, device="cpu"
            )
            for evaluator in handlers:
                evaluator.update(results_merged, outputs)

            # Write to MemmapTensor
            if results_mem is None:
                assert results_prefix is None
                results_prefix, results_mem = self._inference_results_allocate(batch_total * batch_size, results_merged)
            else:
                assert results_prefix is not None
            results_mem[write_index : write_index + samples_in_batch] = results_merged
            write_index += samples_in_batch

            samples_processed += samples_in_batch

            self._event(Event.ON_INFERENCE_STEP, loader=dataloader)

        del results_merged
        del outputs

        # Clear the past memory
        self._past = None

        # Wait for everyone to finish writing
        self._xlr.wait_for_everyone()

        # Run metric computations
        metrics: dict[str, float | str | int | bool] = {}
        visuals: dict[str, pil_image.Image] = {}
        if self._xlr.is_main_process:
            if write_index < batch_total:
                warnings.warn(
                    (f"Expected to process {batch_total} batches, but only processed {write_index}. "),
                    stacklevel=2,
                )
            metrics.update(
                _build_speed_metrics(
                    "inference",
                    t_start_all,
                    sample_amount=samples_processed,
                    step_amount=math.ceil(samples_processed * self._xlr.num_processes),
                )
            )
            assert results_mem is not None, "Expected results memory to be initialized"
            assert results_mem.is_memmap(), "Expected results memory to be memmapped"

            # Run the evaluation handlers on the main process
            for evaluator in handlers:
                _logger.debug(f"Running evaluation handler: {evaluator}")
                assert isinstance(results_mem, TensorDict)

                # Metrics
                handler_metrics = evaluator.compute(results_mem, device=self._xlr.device)
                metrics.update(handler_metrics)

                # Visualizations
                visuals.update(evaluator.plot(results_mem))

            # Store visualizations
            self._store_visualizations(visuals, prefix=prefix)

        # Remove the memmap file
        self._xlr.wait_for_everyone()

        del results_mem
        if self._xlr.is_main_process:
            assert results_prefix is not None
            shutil.rmtree(results_prefix, ignore_errors=True)

        # Store metrics
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{prefix}_jit_compilation_time"] = self.jit_compilation_time  # type: ignore

        # Prefix all keys with prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{prefix}_"):
                metrics[f"{prefix}_{key}"] = metrics.pop(key)

        del dataloader

        return metrics, samples_processed

    def _store_visualizations(self, visuals: dict[str, pil_image.Image], prefix: str = "") -> None:
        """
        Store visualizations that are provided as a mapping of (key) -> (PIL image).
        """

        _logger.info(f"Storing visualizations ({len(visuals)} total)")

        for key, img in visuals.items():
            if self._config.eval_write_visuals:
                img_path = (
                    file_io.Path(self._xlr.project_dir) / "visuals" / f"{prefix}-{self._state.step}" / f"{key}.png"
                )
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img.save(img_path)

            wandb_run = self._xlr.get_tracker("wandb")
            if wandb_run is not None:
                wandb_run.log({key: wandb.Image(img)}, step=self._state.step)

    def _nested_gather(self, tensors: MaybeTensorType, name=None) -> MaybeTensorType:
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return tensors
        if self._xlr.use_distributed:
            tensors = self._xlr.gather(tensors)
        return tensors

    def _inference_step(
        self,
        model: nn.Module,
        inputs: TensorDict,
        ignore_keys: T.Optional[list[str]] = None,
    ) -> ModelOutput:
        """
        Perform an evaluation step on `model` using `inputs`.
        """

        with torch.inference_mode():
            outputs: ModelOutput = model(inputs)
            if self._config.past_index >= 0:
                self._past = outputs[self._config.past_index - 1]
        return outputs


def floating_point_ops(model, inputs: InputType):
    """
    Uses that method to compute the number of floating point
    operations for every backward + forward pass. If using another model, either implement such a method in the
    model or subclass and override this method.

    Args:
        inputs (`TensorDict`):
            The inputs and targets of the model.

    Returns:
        `int`: The number of floating-point operations.
    """
    try:
        flops_fn: T.Callable[[InputType], int] = model.floating_point_ops
    except AttributeError:
        return 0
    return flops_fn(inputs)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
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
