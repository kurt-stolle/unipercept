"""
The `Trainer` class is the main class to handle training and evaluation of any kind of models with any kind of datasets.
"""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import time
import typing as T
from dataclasses import asdict

import accelerate
import safetensors
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from timm.scheduler.scheduler import Scheduler as TimmScheduler
from torch.utils.data import Dataset
from unicore import file_io
from unicore.utils.tensorclass import Tensorclass, TensorDict, TensorDictBase
from uniutils.logutils import get_logger
from uniutils.seed import enable_full_determinism, set_seed

from ._optimizer import OptimizerFactory
from ._scheduler import SchedulerFactory
from ._types import DataLoaderFactory, ModelFactory
from .callbacks import CallbackType
from .callbacks import Delegate as CallbackHandler
from .callbacks import Event, Signal, TrainState, get_checkpoint_name
from .config import TrainConfig
from .debug import DebugMode, DebugUnderflowOverflow
from .memory import MemoryTracker
from .search import SearchBackend, Trial

if T.TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer
    from torch.utils.data import DataLoader

__all__ = ["Trainer"]
__dir__ = __all__

_logger = get_logger(__name__)


def _build_accelerator(config: TrainConfig) -> accelerate.Accelerator:
    from accelerate.accelerator import ProjectConfiguration

    root = file_io.Path(config.root).resolve()

    project_dir = root / "output"
    logging_dir = root / "logs"

    project_dir.mkdir(parents=True, exist_ok=False)
    logging_dir.mkdir(parents=True, exist_ok=False)

    _logger.info("Using directory: %s", str(root))

    acc = accelerate.Accelerator(
        project_dir=project_dir,
        project_config=ProjectConfiguration(
            project_dir=str(project_dir), logging_dir=str(logging_dir), automatic_checkpoint_naming=False
        ),
        step_scheduler_with_optimizer=False,
        log_with="all",
        dispatch_batches=False,
    )

    from pprint import pformat

    _logger.info("Using accelerator: \n%s", pformat(asdict(acc.project_configuration)))
    _logger.info("Current process: %d / %d", acc.process_index, acc.num_processes)

    return acc


InputType: T.TypeAlias = T.Tuple[list[str], TensorDictBase]
TrainingOutputType: T.TypeAlias = T.Dict[str, torch.Tensor]
InferenceOutputType: T.TypeAlias = T.Sequence[T.Dict[str, T.Any]]


class Trainer:
    """
    Training loop
    """

    def __init__(
        self,
        *,
        config: TrainConfig,
        optimizer: OptimizerFactory,
        scheduler: SchedulerFactory,
        callbacks: T.Sequence[CallbackType | type[CallbackType]],
    ):
        self._config: T.Final[TrainConfig] = config
        self._seed()
        self._acc = _build_accelerator(config)

        _logger.info("Initializing Trainer: %s @ %s", config.project_name, config.session_name)

        # Memory metrics - must set up as early as possible
        self._mem_tracker = MemoryTracker(False)
        self._mem_tracker.start()

        # Factory methods
        self._optimizer_factory: T.Final = optimizer
        self._scheduler_factory: T.Final = scheduler

        # HP search
        self._hp_backend: SearchBackend | None = None

        # Callback manager
        self._delegate = CallbackHandler(callbacks)

        # Properties
        self._state = TrainState()
        self._acc.register_for_checkpointing(self._state)
        self._signal = Signal()
        self._flops = 0

        # Event: on_init_end
        self._edge(Event.ON_INIT_END)

        # Stop the memory tracking and update the metrics
        self._mem_tracker.stop_and_update_metrics()

    #
    # Properties proxies to the Accelerator
    #

    @property
    def device(self):
        return self._acc.device

    @property
    def process_count(self):
        return self._acc.num_processes

    @property
    def process_index(self):
        return self._acc.process_index

    def _get_output_dir(self, trial):
        return self._acc.project_dir

    #
    # Seeding
    #

    def _seed(self):
        if self._config.full_determinism:
            enable_full_determinism(self._config.seed)
        else:
            set_seed(self._config.seed)

    #
    # Events
    #

    def _edge(self, event: Event, **kwargs) -> None:
        self._signal = self._delegate(event, self._config, self._state, self._signal, **kwargs)

    #
    # HP tuning
    #

    def _get_trial_name(self, trial: Trial | None = None) -> str:
        if trial is not None:
            return trial.name
        else:
            return "training"

    def _get_trial_params(self, trial: Trial | None = None, hp_backend: SearchBackend | None = None):
        if trial is not None:
            return trial.params
        else:
            return None

    #
    # Training
    #

    def train(
        self,
        model_factory: ModelFactory[Trial, Module],
        loader_factory: DataLoaderFactory[TensorDictBase],
        checkpoint: str | T.Literal["best"] | None = None,
        trial: Trial | None = None,
    ):
        # Throw away the environment now
        self._state.reset()

        # Memory metrics - must set up as early as possible
        self._mem_tracker.start()

        # Trial
        trial_name = self._get_trial_name(trial)
        trial_params = self._get_trial_params(trial, self._hp_backend)

        # Start a new context
        model = model_factory(trial_params)
        optimizer = self._optimizer_factory(model)

        # If 'best' is passed as checking, determine the best checkpoint
        if checkpoint == "best":
            checkpoint = self._find_best_checkpoint(trial_name)
        elif checkpoint == "last":
            checkpoint = self._find_last_checkpoint(trial_name)

        # Create a closure for finding the optimal batch size
        @accelerate.utils.find_executable_batch_size(starting_batch_size=self._config.batch_size)
        def _train_inner(batch_size: int | None = None):
            assert batch_size is not None, "Batch size is not initialized!"

            _logger.info("\n\n" + "*" * 10 + "\n\n")
            _logger.info(f"Using batch size: {batch_size}")

            # Build the loader
            loader = loader_factory(batch_size)

            # Infer the number of steps/updates per epoch
            steps_per_epoch = len(loader)
            updates_per_epoch = math.ceil(steps_per_epoch / self._config.gradient_accumulation_steps)

            # Build the scheduler
            scheduler, train_epochs = self._scheduler_factory(
                optimizer, self._config.get_train_epochs(steps_per_epoch), updates_per_epoch
            )

            # Reset the state
            self._state.reset()
            self._state.trial_name = trial_name
            self._state.trial_params = trial_params
            self._state.logging_steps = self._config.logging_steps
            self._state.eval_steps = self._config.get_eval_interval_steps(steps_per_epoch)
            self._state.save_steps = self._config.get_save_interval_steps(steps_per_epoch)
            self._state.train_steps = self._config.get_train_steps(steps_per_epoch)
            self._state.best_metric = None

            # Load session and run training loop
            return self._train_loop(loader, model, optimizer, scheduler, checkpoint)

        return _train_inner()

    def _find_last_checkpoint(self, trial_name: str) -> str | None:
        checkpoints = [
            path
            for path in file_io.Path(self._acc.project_dir).iterdir()
            if path.is_dir() and self._state.checkpoint_pattern.search(path.as_posix()) is not None
        ]
        if len(checkpoints) == 0:
            return
        return os.path.join(
            self._acc.project_dir,
            max(checkpoints, key=lambda x: int(self._state.checkpoint_pattern.search(x.as_posix()).groups()[0])),
        )

    def _find_best_checkpoint(self, trial_name: str) -> str | None:
        best_path = os.path.join(self._acc.project_dir, get_checkpoint_name(trial_name, "best"))
        if file_io.exists(best_path):
            return best_path
        return None

    def _init_loggers(self):
        n = self._state.trial_name
        assert n is not None, "Trial name is not initialized!"
        p = self._state.trial_params
        self._acc.init_trackers(n, config=p)

    def _train_loop(
        self,
        loader: DataLoader,
        model: Module,
        optimizer: Optimizer,
        scheduler: TimmScheduler,
        checkpoint: str | None = None,
    ):
        _logger.debug("Creating new session...")

        # Register and prepare using Accelerate
        self._acc.free_memory()

        loader, model, scheduler, optimizer = self._acc.prepare(loader, model, scheduler, optimizer)
        # NOTE: the state object is registered in `__init__` and is not re-registered here

        # Restore the state
        if checkpoint is not None:
            _logger.debug("Loading checkpoint: %s", checkpoint)
            self._acc.load_state(checkpoint)
        else:
            _logger.debug("No checkpoint found, starting from scratch...")

        # Initialize loggers
        self._init_loggers()

        # Debugging
        debug_overflow = DebugUnderflowOverflow(model)  # noqa

        # Variables that track the progress of the training
        time_start = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Create a loss tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0

        self._edge(Event.ON_TRAIN_BEGIN)

        total_batched_samples = 0

        steps_per_epoch = len(loader)
        start_epoch = math.floor(self._state.epoch)
        steps_trained_in_current_epoch = int((self._state.epoch - start_epoch) * steps_per_epoch)
        train_epochs = math.ceil(self._state.train_steps / steps_per_epoch)

        # Check if the loader requires an epochs state
        if hasattr(loader, "epoch"):
            loader.epoch = start_epoch
        if hasattr(loader.sampler, "epoch"):
            loader.sampler.epoch = epoch

        # -- Epoch outer loop ------------------------------------------------------------------------------------------

        for epoch in range(start_epoch, train_epochs):
            # Set the epoch iterator to the original dataloader
            epoch_iterator = loader

            # Reset the past mems state at the beginning of each epoch if necessary.
            # if config.past_index >= 0:
            #     self._past = None

            self._edge(Event.ON_EPOCH_BEGIN)

            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = self._acc.skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0

            steps_in_epoch = len(epoch_iterator)

            # -- Epoch inner loop --------------------------------------------------------------------------------------

            step = -1  # If the value of the iterator is still -1 after the loop, something went wrong
            for step, inputs in enumerate(epoch_iterator):
                inputs: InputType
                assert (
                    isinstance(inputs, T.Sequence) and len(inputs) == 2
                ), f"The dataset should return a tuple in the form (ids, tensordict, ...), got {type(inputs)}"
                assert isinstance(inputs[0], T.Sequence) and isinstance(
                    inputs[1], (TensorDictBase, Tensorclass)
                ), f"The dataset should return a tuple in the form (ids, tensordict, ...), got ({','.join([type(i).__name__ for i in inputs])})"

                total_batched_samples += 1

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
                    self._edge(Event.ON_STEP_BEGIN)

                with self._acc.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if self._config.logging_nan_inf_filter and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self._state.step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self._flops += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= self._config.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % self._config.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # Gradient clipping
                    if self._config.max_grad_norm is not None and self._config.max_grad_norm > 0:
                        if hasattr(optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            optimizer.clip_grad_norm(self._config.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(self._config.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer),
                                self._config.max_grad_norm,
                            )
                        else:
                            self._acc.clip_grad_norm_(
                                model.parameters(),
                                self._config.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer.step()
                    if not self._acc.optimizer_step_was_skipped:
                        scheduler.step_update(epoch)  # TODO metric is not used
                    optimizer.zero_grad()
                    self._state.step += 1
                    self._state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self._signal = self.callbacks.on_step_end(config, state, self._signal)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self._signal = self.callbacks.on_substep_end(config, state, self._signal)

                if self._signal.should_epoch_stop or self._signal.should_training_stop:
                    break

            # -- End of epoch ------------------------------------------------------------------------------------------
            if step < 0:
                _logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self._signal.should_training_stop = True

            self._edge(Event.ON_EPOCH_END)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self._signal.should_training_stop:
                break

        # -- End of training -------------------------------------------------------------------------------------------

        if config.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        _logger.info("\n\nTraining completed.\n\n")

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / state.global_step

        # Compute flops
        self._state.total_flops += self._flops
        self._flops = 0

        # Compute runtime
        time_train = time.time() - time_start

        # Report final metrics
        metrics = {}
        metrics["train_runtime"] = round(time_train, 4)
        metrics["train_samples_per_second"] = round(num_train_samples / time_train, 3)
        metrics["train_steps_per_second"] = round(num_train_steps / time_train, 3)
        metrics["total_flops"] = self._state.total_flops
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._mem_tracker.stop_and_update_metrics(metrics)
        self._log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.context.should_save
            and state.best_model_checkpoint is not None
            and self.self.context.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, state.best_model_checkpoint):
                    _logger.info(f"Deleting older checkpoint [{checkpoint}] due to self.context.save_total_limit")
                    shutil.rmtree(checkpoint)

        self._edge(Event.ON_TRAIN_END)

        return self._ctx.model

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self._signal.should_log:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self._state.step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self._state.step
            self.store_flops()

            self._log(logs)

        metrics = None
        if self._signal.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self._state.step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self._config.metric
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self._signal.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self._edge(Event.ON_SAVE)

    def hyperparameter_search(
        self,
        model_factory: T.Callable[[Trial], Module],
        hp_space: T.Optional[T.Callable[[Trial], dict[str, float]]] = None,
        compute_objective: T.Optional[T.Callable[[dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str | list[str] = "minimize",
        **kwargs,
    ) -> T.Any:
        # TODO: merge
        raise NotImplementedError

    def _log(self, logs: dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`dict[str, float]`):
                The values to log.
        """
        if self._state.epoch is not None:
            logs["epoch"] = round(self._state.epoch, 2)

        output = {**logs, **{"step": self._state.step}}
        self._state.log_history.append(output)
        self._signal = self._delegate.on_log(self._config, self._state, self._signal, logs)

    def training_step(self, model: Module, inputs: InputType) -> torch.Tensor:
        """
        A single training step (forward + backward + update).
        """
        model.train()
        loss = self.compute_loss(model, inputs)
        self._acc.backward(loss)

        return loss.detach() / self._config.gradient_accumulation_steps

    def compute_loss(self, model: Module, inputs: InputType):
        """
        Compute the loss on a batch of inputs.
        """
        outputs = model(*inputs)

        if isinstance(outputs, torch.Tensor):
            loss = outputs
        elif isinstance(outputs, T.Mapping):
            # Sum each value
            losses = [v for k, v in outputs.items() if k.startswith("loss")]
            assert all([isinstance(l, torch.Tensor) for l in losses]), "Losses must be tensors"
            loss = torch.mean(torch.stack(losses))
        else:
            raise ValueError(f"Unexpected type of `outputs`: {type(outputs)}")

        return loss

    def _save(self, output_dir: T.Optional[str] = None, state_dict=None):
        # TODO: Save a model checkpoint
        pass

    def _sorted_checkpoints(self, use_mtime=False) -> T.Iterator[str]:
        output_dir = file_io.Path(self._acc.project_dir)
        ordering_and_checkpoint_path: list[tuple[float | int, file_io.Path]] = []
        glob_checkpoints = (str(x) for x in output_dir.glob(f"{self._state.checkpoint_prefix}-*") if os.path.isdir(x))

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{self._state.checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

        # TODO: copy best checkpoint

        yield from map(str, checkpoints_sorted)

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self._config.save_total_limit is None or self._config.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = list(self._sorted_checkpoints(use_mtime=use_mtime))
        if len(checkpoints_sorted) <= self._config.save_total_limit:
            return

        # If save_total_limit=1 with load_best_model_at_end=True, we could end up deleting the last checkpoint, which
        # we don't do to allow resuming.
        save_total_limit = self._config.save_total_limit

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            _logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
            shutil.rmtree(checkpoint, ignore_errors=True)

    def evaluate(
        self,
        model_factory: ModelFactory[Trial, Module],
        loader_factory: DataLoaderFactory[TensorDictBase],
        checkpoint: str | T.Literal["best"] | None = None,
    ) -> dict[str, float]:
        # Memory metrics - must set up as early as possible
        self._mem_tracker.start()

        loader = loader_factory(self._acc.num_processes)
        start_time = time.time()

        output = self.evaluation_loop(
            loader,
            description="Evaluation",
        )

        total_batch_size = self._config.eval_batch_size * self._config.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._log(output.metrics)
        self._edge(Event.ON_EVALUATE, metrics=output.metrics)
        self._mem_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(self, dataset: Dataset, output_path: str | file_io.Path | None = None) -> PredictionOutput:
        # Memory metrics - must set up as early as possible
        self._mem_tracker.start()

        loader = torch.utils.data.DataLoader(dataset)
        start_time = time.time()

        output = self.evaluation_loop(loader, description="Prediction")
        total_batch_size = self._config.eval_batch_size * self._config.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self._signal = self._delegate.on_predict(self._config, self._state, self._signal, output.metrics)
        self._mem_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: T.Optional[bool] = None,
        ignore_keys: T.Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self._config

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self._acc._models) == 0 and model is self.model:
            model = (
                self._acc.prepare(model)
                if self.is_deepspeed_enabled
                else self._acc.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self._config.eval_batch_size

        _logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            _logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            _logger.info("  Num examples: Unknown")
        _logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self._delegate.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._acc.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self._acc.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._acc.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self._acc.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._acc.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self._acc.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self._acc.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self._signal = self._delegate.on_prediction_step(args, self._state, self._signal)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self._acc.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _nested_gather(self, tensors, name=None):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        if (self._config.distributed_state is not None and self._config.distributed_state.distributed_type != "NO") or (
            self._config.distributed_state is None and self._config.local_rank != -1
        ):
            tensors = distributed_concat(tensors)
        return tensors

    def prediction_step(
        self,
        model: Module,
        inputs: TensorDictBase,
        ignore_keys: T.Optional[list[str]] = None,
    ) -> InferenceOutputType:
        """
        Perform an evaluation step on `model` using `inputs`.
        """

        with torch.inference_mode():
            outputs: InferenceOutputType = model(*inputs)
            if isinstance(outputs, dict):
                outputs = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                outputs = outputs
            if self._config.past_index >= 0:
                self._past = outputs[self._config.past_index - 1]

        # logits = self._acc.nested_detach(logits)
        return outputs

    def floating_point_ops(self, inputs: TensorDictBase):
        """
        Uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`TensorDictBase`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if hasattr(self.model, "floating_point_ops"):
            return self.model.floating_point_ops(inputs)
        else:
            return 0

    def _gather_and_numpify(self, tensors, name):
        """
        Gather value of `tensors` (tensor or list/tuple of nested tensors) and convert them to numpy before
        concatenating them to `gathered`
        """
        if tensors is None:
            return
        elif self._config.parallel_mode == ParallelMode.DISTRIBUTED:
            tensors = distributed_concat(tensors)

        return nested_numpify(tensors)
