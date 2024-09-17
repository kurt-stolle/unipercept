"""
The `Engine` class is the main class to handle training and evaluation of any kind of models with any kind of datasets.
"""

from __future__ import annotations

import concurrent.futures
import enum as E
import functools
import gc
import math
import operator
import re
import shutil
import sys
import time
import typing as T
from datetime import datetime
from typing import override

import torch
import torch.optim
import torch.types
import torch.utils.data
from omegaconf import DictConfig, OmegaConf
from PIL import Image as pil_image
from tensordict import TensorDict, TensorDictBase, pad_sequence
from torch import Tensor, nn

from unipercept import file_io
from unipercept.data import DataLoaderFactory
from unipercept.engine._params import (
    EngineParams,
    EvaluationSuite,
    TrainingStage,
)
from unipercept.engine._trial import Trial, TrialWithParameters
from unipercept.engine.accelerate import Accelerator, find_executable_batch_size
from unipercept.engine.callbacks import CallbackType, Delegate, Event, Signal, State
from unipercept.engine.debug import DebugMode, DebugUnderflowOverflow
from unipercept.engine.memory import MemoryTracker
from unipercept.engine.writer import ResultsWriter
from unipercept.log import create_table, logger
from unipercept.model import InputData, ModelBase, ModelOutput
from unipercept.state import (
    barrier,
    check_main_process,
    get_process_count,
    get_process_index,
    get_total_batchsize,
    reduce,
)
from unipercept.types import Pathable
from unipercept.utils.memory import retry_if_cuda_oom
from unipercept.utils.seed import set_seed
from unipercept.utils.status import StatusDescriptor
from unipercept.utils.tensorclass import Tensorclass, is_tensordict_like
from unipercept.utils.time import ProfileAccumulator, profile
from unipercept.utils.ulid import ULID

if T.TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from accelerate.optimizer import AcceleratedOptimizer
    from timm.scheduler.scheduler import Scheduler as TimmScheduler

    import unipercept as up
    from unipercept.evaluators import Evaluator as Evaluator
    from unipercept.model import ModelFactory

MaybeTensorType = T.TypeVar("MaybeTensorType", bound=torch.Tensor | None)

__all__ = ["Engine", "EngineStatus"]

InputType: T.TypeAlias = TensorDict | Tensorclass
TrainingOutputType: T.TypeAlias = dict[str, torch.Tensor]
InferenceOutputType: T.TypeAlias = T.Sequence[dict[str, T.Any]]


class EngineStatus(E.IntFlag):
    """
    The current status of the engine. This status is not part of the persistent/stored state. Used internally for
    control flow, e.g. in cases where the evaluation loop is ran while also training.
    """

    IS_TRAINING_PROCEDURE = E.auto()
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
        "session_id",
        "find_batch_size",
        "_config",
        "_dataloaders",
        "_evaluators",
        "_stages",
        "__dict__",
    )

    def __init__(
        self,
        *,
        params: up.engine.EngineParams,
        callbacks: T.Sequence[CallbackType | type[CallbackType]],
        stages: T.Iterable[up.engine.TrainingStage] | None = None,
        evaluators: T.Mapping[str, EvaluationSuite] | None = None,
        log_events: bool = False,
        find_batch_size: bool = False,
        **kwargs,
    ):
        self._default_setup()

        self.session_id = _generate_session_id()
        self._mem_tracker = MemoryTracker(enabled=not params.memory_tracker)
        self._mem_tracker.start("init")  # must set up as early as possible

        self._params: T.Final[EngineParams] = params
        self._seed()
        self._state = State()
        self._xlr = None
        self._root = None
        self._config = None
        self.find_batch_size = find_batch_size

        if find_batch_size:
            logger.warning(
                "Batch size finding is enabled. This should not be used in production, "
                "for final experiments, or in distributed settings!"
            )

        self._stages = list(stages) if stages is not None else []
        self._evaluators: T.Final[dict[str, EvaluationSuite]] = (
            {k: v for k, v in evaluators.items()} if evaluators is not None else {}
        )
        self._signal = Signal()
        self._delegate = Delegate(callbacks, verbose=log_events)
        self._flops = 0
        self._step_last_logged = -1
        self._step_last_saved = -1
        self._step_last_evaluated = -1
        self._recover_path = None  # See: `recover` method

        # Handle keyword arguments
        if kwargs.pop("grad_norm_smoother", None) is not None:
            logger.warning(
                "The `grad_norm_smoother` keyword is no longer supported and will "
                "be removed in a future version. "
                "Use `callbacks.GradientClippingCallback` instead."
            )

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
        self._edge(Event.ON_ACCELERATOR_SETUP, accelerator=xlr)

        self._xlr = xlr
        return xlr

    def select_inputs(self, model: nn.Module, inputs: InputType) -> tuple[T.Any]:
        """
        Run the `select_inputs` method on the unwrapped model (if it exists) and
        return the result.

        Parameters
        ----------
        model
            Model, potentialy wrapped by an adapter or accelerator.
        inputs
            Input data object.

        Returns
        -------
            Tuple to be passed as *args to the model
        """
        model = self.xlr.unwrap_model(model)
        if hasattr(model, "select_inputs"):
            return tuple(model.select_inputs(inputs, device=self.xlr.device))
        return (inputs,)

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
        from unipercept.config.lazy import load_config_local

        if self._config is not None:
            return self._config

        path = self.config_path
        if not path.exists():
            msg = f"Could not find configuration file at {path!r}"
            raise FileNotFoundError(msg)

        logger.info("Loading configuration from %s", path)

        try:
            lazy = load_config_local(str(path))
        except Exception as e:  # noqa: PIE786
            msg = f"Could not load configuration from {path!r} {e}"
            logger.warning(msg)
            return {}

        lazy_obj = OmegaConf.to_container(lazy, resolve=False)
        assert isinstance(lazy_obj, dict)

        self._config = lazy_obj

        return T.cast(dict[str, T.Any], lazy_obj)

    @config.setter
    def config(self, value: DictConfig) -> None:
        from unipercept.config.lazy import save_config

        if check_main_process():
            path = self.config_path
            if path.exists():
                msg = f"Configuration file already exists at {path}"
                raise FileExistsError(msg)
            logger.info("Saving configuration to %s", path)
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
            self._edge(Event.ON_MODEL_SETUP, model=model, training=False)
            self.xlr.prepare_model(model, evaluation_mode=True).to(self.xlr.device)

        if checkpoint is not None:
            self._recover_path = str(checkpoint)
            self._load_state(self._recover_path)  # type: ignore

        logger.info("Recovered engine state at step %d", self._state.step)

    @status(EngineStatus.IS_TRAINING_PROCEDURE)
    def run_training_procedure(
        self, model_factory, start_stage: int, *, weights: str | None = None
    ):
        """
        Run the training procedure for a specific stage. This method is called by the `train` method.
        """

        logger.info(
            "Starting training procedure:\n%s",
            create_table(
                {"starting stage": start_stage, "initial weights": weights},
                format="long",
            ),
        )

        stage_num = start_stage
        while True:
            weights = self.run_training(model_factory, stage=stage_num, weights=weights)
            print("\n\n", file=sys.stderr, flush=True)
            print("\n\n", file=sys.stderr, flush=True)
            stage_num += 1
            if stage_num >= len(self._stages):
                break
            logger.info(
                "Training completed for stage %d. Moving to next...", stage_num - 1
            )

        self._stop_experiment_trackers()
        logger.info(
            "Training completed for all stages: \n%s",
            create_table(
                {"final stage": stage_num - 1, "final weights": weights}, format="long"
            ),
        )

    def get_training_stage(self, num: int = -1) -> TrainingStage:
        return self._stages[num]

    @status(EngineStatus.IS_TRAINING_RUN)
    def run_training(
        self,
        model_factory: ModelFactory,
        *,
        trial: Trial | None = None,
        stage: int | TrainingStage | None = None,
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
            stage = self.get_training_stage(stage_num)
        elif isinstance(stage, int):
            if stage < 0 or stage >= len(self._stages):
                raise ValueError(
                    f"Stage {stage} is out of bounds. This engine has {len(self._stages)} stages, "
                    "and a value of -1 could indicate that the stage was recovered from a checkpoint "
                    "that used a custom StageDefinition instead of a number."
                )
            stage_num = stage
            stage = self.get_training_stage(stage)
        else:
            try:
                stage_num = self._stages.index(stage)
            except ValueError:
                stage_num = -1

        self._state.stage = stage_num
        logger.info(f"Training run: stage {stage_num}...")

        if not isinstance(stage, TrainingStage):
            raise TypeError(
                f"Expected stage to be of type TrainingStage, got {type(stage)}"
            )

        trial = TrialWithParameters(
            name="stage_" + str(stage_num),
            config=stage.model_config,
            weights=weights,
            parent=trial,
        )

        self._start_experiment_trackers(restart=False)

        def train_inner(batch_size: int) -> nn.Module:
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
            if batch_size >= stage.batch_size:
                logger.debug("Training start: batch size %d", batch_size)
            else:
                logger.debug(
                    "Training restart: batch size %d (original: %d)",
                    batch_size,
                    stage.batch_size,
                )

            gradient_accumulation = stage.gradient_accumulation * (
                stage.batch_size // batch_size
            )
            # gradient_accumulation = 1  # PyTorch 2.2: broken
            assert (
                gradient_accumulation > 0
            ), "Expected gradient accumulation to be greater than 0"

            loader, steps_per_epoch, updates_per_epoch = self.build_training_dataloader(
                stage.loader, batch_size, gradient_accumulation
            )

            model = model_factory(
                overrides=trial.config,
                weights=trial.weights,
            )
            scheduled_epochs = stage.get_epochs(steps_per_epoch)
            assert (
                scheduled_epochs > 0
            ), "Expected scheduled epochs to be greater than 0"

            extra_params = []
            self._edge(Event.ON_OPTIMIZER_SETUP, stage=stage, extra_params=extra_params)

            optimizer = stage.optimizer(
                model, stage.batch_size, extra_params=extra_params
            )
            if stage.scheduler is not None:
                scheduler, train_epochs = stage.scheduler(
                    optimizer, scheduled_epochs, updates_per_epoch
                )
            else:
                scheduler = None
                train_epochs = float(scheduled_epochs)
            assert train_epochs > 0, "Expected train epochs to be greater than 0"

            logger.debug(
                "Training start: running %.1f epochs (%d steps)",
                train_epochs,
                int(train_epochs * steps_per_epoch),
            )

            # Reset the state
            self._state.register_training(
                logging_steps=self._params.logging_steps,
                eval_steps=self._params.get_eval_interval_steps(steps_per_epoch),
                save_steps=self._params.get_save_interval_steps(steps_per_epoch),
                train_steps=stage.get_steps(steps_per_epoch),
                gradient_accumulation=gradient_accumulation,
                best_metric=None,
                trial_name=trial.name,
                trial_config=trial.config,
            )

            return self.run_training_loop(
                loader,
                model,
                optimizer,
                scheduler,
                trial=trial,
            )

        # Add FINDING_BATCH_SIZE flag to status
        self.status |= EngineStatus.FINDING_BATCH_SIZE

        if self.find_batch_size:
            if self.xlr.use_distributed:
                msg = "Batch size finding is not supported in distributed settings!"
                raise RuntimeError(msg)
            train = find_executable_batch_size(starting_batch_size=stage.batch_size)(
                train_inner
            )
        else:
            train = functools.partial(train_inner, stage.batch_size)

        result = train()

        # If we are not running a larger procedure of stages, stop trackers
        if EngineStatus.IS_TRAINING_PROCEDURE not in self.status:
            self._stop_experiment_trackers()

        return self._save_weights(None, result)

    def list_evaluation_suites(self) -> list[str]:
        return [k for k, v in self._evaluators.items() if v.enabled]

    def get_evaluation_suite(self, key: str) -> EvaluationSuite:
        try:
            return self._evaluators[key]
        except KeyError:
            msg = f"Evaluation suite {key} not found. Available suites: {list(self._evaluators.keys())}"
            raise KeyError(msg)

    @status(EngineStatus.IS_EVALUATION_RUN)
    @torch.no_grad()
    # @torch.inference_mode()
    def run_evaluation(
        self,
        model_factory: ModelFactory | None,
        trial: Trial | None = None,
        *,
        suites: T.Collection[str] | None = None,
        prefix: str = "evaluation",
        path: Pathable | None = None,
        optimizer: AcceleratedOptimizer | None = None,
    ) -> dict[str, float]:
        logger.info("Starting evaluation procedure...")

        metrics_overall = {}

        if suites is None:
            suites = self.list_evaluation_suites()
        else:
            for k in suites:
                if k not in self._evaluators:
                    msg = f"Evaluation suite {k} is not available"
                    raise ValueError(msg)

        for suite_key in suites:
            suite = self._evaluators[suite_key]
            logger.info(
                "Running inference on %s for %d handlers",
                repr(suite),
                len(suite.handlers),
            )

            prefix_suite = "/".join([prefix, suite.name])

            torch.cuda.empty_cache()
            gc.collect()
            # Memory metrics - must set up as early as possible
            self._mem_tracker.start("eval")

            if model_factory is not None:
                assert callable(model_factory), "Expected model factory to be callable"
                assert not isinstance(model_factory, nn.Module), (
                    "Expected model factory to be a callable that returns a model, "
                    "not the model itself"
                )
                if trial is not None:
                    model = model_factory(
                        overrides=trial.config,
                        weights=trial.weights,
                    )
                else:
                    model = model_factory()
            else:
                model = None

            loader = suite.loader(suite.batch_size, use_distributed=True)

            if path is not None:
                path_suite = file_io.Path(path) / suite.name
                path_suite.mkdir(parents=True, exist_ok=True)
            else:
                path_suite = None

            metrics, samples_processed = self.run_inference_loop(
                model,
                loader,
                prefix=prefix_suite,
                handlers=suite.handlers,
                path=path_suite,
                optimizer=optimizer,
            )

            self._edge(Event.ON_EVALUATE, metrics=metrics)
            self._mem_tracker.stop_and_update_metrics("eval", metrics)

            del loader
            del model

            for metric_key in list(metrics.keys()):
                if not metric_key.startswith(prefix_suite):
                    metrics[prefix_suite] = metrics.pop(metric_key)
            metrics_overall.update(metrics)

        if len(metrics_overall) == 0:
            logger.warning("No metrics were logged during evaluation")

        self._push_logs(metrics_overall)

        return metrics_overall

    @T.overload
    def build_training_dataloader(
        self,
        dataloader: DataLoaderFactory,
        batch_size: int,
        gradient_accumulation: None = None,
    ) -> tuple[torch.utils.data.DataLoader, int, None]: ...

    @T.overload
    def build_training_dataloader(
        self,
        dataloader: DataLoaderFactory,
        batch_size: int,
        gradient_accumulation: int,
    ) -> tuple[torch.utils.data.DataLoader, int, int]: ...

    def build_training_dataloader(
        self,
        dataloader: DataLoaderFactory,
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
        dl = dataloader(batch_size // get_process_count(), use_distributed=False)
        steps_per_epoch = len(dl) // get_process_count()

        if gradient_accumulation is not None:
            updates_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation)
        else:
            updates_per_epoch = None

        # Tabulate and log the loader information
        logger.debug(
            "Using dataloader settings:\n%s",
            create_table(
                {
                    "batch size": batch_size,
                    "batch count": len(dl),
                    "gradient acc.": gradient_accumulation,
                    "processes": get_process_count(),
                    "steps/epoch": steps_per_epoch,
                    "updates/epoch": updates_per_epoch,
                }
            ),
        )

        return dl, steps_per_epoch, updates_per_epoch

    def run_training_step(
        self,
        model: ModelBase,
        inputs: InputType,
        *,
        optimizer: AcceleratedOptimizer | None = None,
    ) -> dict[str, Tensor]:
        """
        A single training step (forward + backward + update).
        """
        model.train()
        if optimizer is not None:
            try:
                optimizer.train()
            except AttributeError:
                pass

        args = self.select_inputs(model, inputs)
        outputs = _forward(model, args)
        assert outputs.losses is not None

        loss_tensor = torch.stack([v for v in outputs.losses.values()])
        if not loss_tensor.isfinite().all():
            loss_nonfinite = ", ".join(
                [
                    f"{k}={v.detach().item()}"
                    for k, v in outputs.losses.items()
                    if not torch.isfinite(v).all()
                ]
            )
            logger.warning("Loss contains non-finite values: %s", loss_nonfinite)
            loss_tensor = torch.nan_to_num(loss_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        if self._params.train_sum_losses:
            self.xlr.backward(loss_tensor.sum())
        else:
            self.xlr.backward(loss_tensor, gradient=torch.ones_like(loss_tensor))

        loss_logs = {
            k: v.detach() / self._state.gradient_accumulation
            for k, v in outputs.losses.items()
        }
        loss_logs["total"] = torch.stack(list(loss_logs.values())).sum()

        return loss_logs

    def run_training_loop(
        self,
        loader: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: TimmScheduler | None,
        **kwargs,
    ) -> nn.Module:
        """
        The main training loop. This method is called by the `train` method.
        """

        # Backend configuration
        self.xlr.gradient_accumulation_steps = self._state.gradient_accumulation
        self.xlr.free_memory()

        # Sync backnorm
        if self._params.convert_sync_batchnorm:
            logger.debug("Train loop: converting BatchNorm to SyncBatchNorm")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self._edge(Event.ON_MODEL_SETUP, model=model, training=True)
        model = self.xlr.prepare_model(model).to(self.xlr.device)

        if scheduler is not None:
            loader, optimizer, scheduler = self.xlr.prepare(
                loader, optimizer, scheduler
            )
        else:
            loader, optimizer = self.xlr.prepare(loader, optimizer)

        # First load the initial weights, then the state
        try:
            self._load_state(None)  # type: ignore
        except FileNotFoundError:
            logger.debug("Train loop: no previous state found")

        # Debugging
        # debug_overflow = DebugUnderflowOverflow(model)  # noqa
        if DebugMode.UNDERFLOW_OVERFLOW & self._params.debug:
            logger.debug("Train loop: underflow/overflow debugging is enabled")
            DebugUnderflowOverflow(model)
        else:
            logger.debug("Train loop: underflow/overflow debugging is disabled")

        # Variables that track the progress of the training
        time_start = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Create tensor to store the losses
        tr_loss: dict[str, Tensor] | None = None
        self._edge(
            Event.ON_TRAIN_BEGIN, model=model, optimizer=optimizer, scheduler=scheduler
        )

        total_session_samples = 0
        total_session_steps = 0

        steps_per_epoch = len(loader)
        start_epoch = math.floor(self._state.epoch)
        steps_trained_in_current_epoch = int(
            (self._state.epoch - start_epoch) * steps_per_epoch
        )
        train_epochs = math.ceil(self._state.train_steps / steps_per_epoch)

        # Check if the loader requires an epochs state
        if hasattr(loader, "epoch"):
            loader.epoch = start_epoch
        if hasattr(loader.sampler, "epoch"):
            loader.sampler.epoch = start_epoch

        for epoch in range(start_epoch, train_epochs):
            # Set the epoch iterator to the original dataloader
            epoch_iterator = loader

            self._edge(
                Event.ON_TRAIN_EPOCH_BEGIN,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                logger.debug(
                    "Train loop: skipping the first %d steps",
                    steps_trained_in_current_epoch,
                )

                epoch_iterator = self.xlr.skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0

            steps_in_epoch = len(epoch_iterator)
            epoch_enumerator = enumerate(epoch_iterator)
            timings_accumulator = ProfileAccumulator()
            for timings in timings_accumulator.run(warmup=10):
                with profile(timings, "load_batch") as profile_handle:
                    try:
                        step, data = next(epoch_enumerator)
                    except StopIteration:
                        profile_handle.close()
                        break
                    total_session_samples += 1

                    sources, inputs = data

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        profile_handle.close()
                        continue
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                if step % self._state.gradient_accumulation == 0:
                    self._edge(
                        Event.ON_TRAIN_STEP_BEGIN,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                    )

                with profile(timings, "step"):
                    with self.xlr.accumulate(model):
                        tr_loss_step = self.run_training_step(
                            model,
                            inputs,
                            optimizer=optimizer,
                        )

                # Add the losses individually
                if total_session_steps == 0:
                    # If this is the first step in the current session, the tensordict keys have not yet been
                    # initialized.
                    assert tr_loss is None
                    tr_loss = {
                        k: torch.tensor(0.0, device=self.xlr.device)
                        for k in tr_loss_step.keys()
                    }  # type: ignore
                else:
                    assert tr_loss is not None

                for k, tr_loss_value in tr_loss.items():
                    tr_loss_step_value = tr_loss_step.get(
                        k, torch.tensor(torch.nan, device=tr_loss_value.device)
                    )
                    if self._params.logging_nan_inf_filter and (
                        torch.isnan(tr_loss_step_value)
                        or torch.isinf(tr_loss_step_value)
                    ):
                        tr_loss_value += tr_loss_value / (
                            1 + self._state.step - self._step_last_logged
                        )  # type: ignore
                    else:
                        tr_loss_value += tr_loss_step_value

                # Compute flops
                self._flops += float(_flops(model, inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= self._state.gradient_accumulation
                    and (step + 1) == steps_in_epoch
                )

                with profile(timings, "optimize"):
                    if (
                        total_session_samples % self._state.gradient_accumulation == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        if self.xlr.sync_gradients:
                            self.xlr.unscale_gradients()
                            self._edge(
                                Event.ON_TRAIN_GRADIENTS, model=model, losses=tr_loss
                            )

                        optimizer.step()
                        if not self.xlr.optimizer_step_was_skipped:
                            if scheduler is not None:
                                scheduler.step_update(self._state.step, metric=None)
                        else:
                            pass
                            # logger.debug("Step was skipped")
                        optimizer.zero_grad()
                        self._state.register_step(
                            epoch=epoch,
                            step=step,
                            steps_skipped=steps_skipped,
                            steps_in_epoch=steps_in_epoch,
                        )
                        self._edge(
                            Event.ON_TRAIN_STEP_END,
                            model=model,
                            optimizer=optimizer,
                            losses=tr_loss,
                        )
                        self._train_handle_signals(
                            tr_loss, model, optimizer, timings_accumulator, **kwargs
                        )
                    else:
                        self._edge(
                            Event.ON_TRAIN_SUBSTEP_END,
                            step_last_logged=self._step_last_logged,
                        )

                total_session_steps += 1

                if self._signal.should_epoch_stop or self._signal.should_training_stop:
                    logger.debug(
                        f"Stopping epoch @ step {step} due to signal {self._signal}"
                    )
                    break

            if tr_loss is None:
                logger.warning("Epoch was ended without running any steps.")
                self._signal.should_training_stop = True
                tr_loss = {}
            else:
                assert tr_loss is not None

            if scheduler is not None:
                scheduler.step(round(self._state.epoch), metric=None)

            self._edge(Event.ON_TRAIN_EPOCH_END)
            self._train_handle_signals(
                tr_loss, model, optimizer, timings_accumulator, **kwargs
            )

            epochs_trained += 1

            if self._signal.should_training_stop:
                break

        logger.info("Training completed (stage %d).", self._state.stage)

        self._signal.should_save = True
        self._signal.should_evaluate = True
        self._train_handle_signals(None, model, optimizer, None, **kwargs)

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
        self._push_logs(metrics)
        self._edge(Event.ON_TRAIN_END)

        return model

    # @torch.inference_mode() # DDP issues!
    def run_inference_step(
        self,
        model: nn.Module,
        inputs: TensorDict,
        optimizer: AcceleratedOptimizer | None = None,
    ) -> TensorDictBase:
        """
        Perform an evaluation step on `model` using `inputs`.
        """
        model = model.to(self.xlr.device)
        model.eval()

        if optimizer is not None:
            try:
                optimizer.eval()
            except AttributeError:
                pass

        inputs_device = inputs.device or torch.device("cpu")
        inputs_shape = inputs.batch_size
        inputs = inputs.to(self.xlr.device)
        if len(inputs_shape) == 0:
            inputs = inputs.unsqueeze(0)
        elif len(inputs_shape) != 1:
            msg = f"Expected inputs to have a batch dimension, got {inputs_shape}"
            raise ValueError(msg)

        args = self.select_inputs(model, inputs)
        outputs: ModelOutput = _forward(model, args)
        assert outputs.predictions is not None

        preds = outputs.predictions
        if isinstance(preds, list):
            list_type = type(next(iter(preds)))
            assert all(isinstance(v, list_type) for v in preds)
            if not is_tensordict_like(list_type):
                preds = [TensorDict(v, batch_size=[]) for v in outputs.predictions]
            preds = pad_sequence(preds, pad_dim=0, return_mask=True)
            preds.rename_key_("masks", "valid")
        elif not is_tensordict_like(preds):
            preds = TensorDict(preds, batch_size=inputs_shape)

        if len(inputs_shape) == 0:
            preds = preds[0]  # type: ignore

        preds = preds.to(inputs_device)

        return T.cast(TensorDictBase, preds)

    @status.assert_status(
        ~(EngineStatus.IS_TRAINING_RUN | EngineStatus.IS_EVALUATION_RUN)
    )
    @status(EngineStatus.IS_PREDICTION_RUN)
    @torch.no_grad()
    def predict(
        self,
        model: nn.Module,
        data: torch.utils.data.DataLoader | T.Iterable[InputData],
        *,
        prefix: str = "pred",
    ) -> TensorDict:
        raise NotImplementedError("TODO: Implement prediction")

    # @torch.inference_mode()  # DDP issues!
    def run_inference_loop(
        self,
        model: nn.Module | None,
        dataloader: torch.utils.data.DataLoader,
        prefix: str,
        handlers: T.Mapping[str, Evaluator],
        path: file_io.PathLike | None = None,
        *,
        weights: file_io.PathLike | None = None,
        optimizer: AcceleratedOptimizer | None = None,
    ) -> tuple[dict[str, T.Any], int]:
        """
        Evaluation loop, which roughly follows the folowing procedure:

            (1) prepare the model and data loader
            (2) run prediction on the dataset and feed results through each evaluator's preprocessing function
            (3) iterate the dataset again, and run the evaluation function of each evaluator
        """

        logger.info("Starting inference procedure...")
        self._start_experiment_trackers(restart=False)

        torch.cuda.empty_cache()
        gc.collect()

        # Find the size of the dataset
        batch_size = dataloader.batch_size
        if batch_size is None:
            raise RuntimeError("Batch size must be set on the dataloader")
        batch_total, batch_offsets = get_total_batchsize(dataloader, self.xlr.device)
        samples_total = batch_total * batch_size
        logger.debug(
            f"Expecting {samples_total} samples ({batch_total} batches, offsets {batch_offsets})"
        )

        # Prepare model
        if model is not None:
            if weights is not None:
                model = self._load_weights(weights, model)

            # TODO: The FP32 wrapper added by prepare breaks our model
            if self.xlr.unwrap_model(model) is model:
                logger.debug(
                    "Inference loop: preparing model (%s)", model.__class__.__name__
                )
                self._edge(Event.ON_MODEL_SETUP, model=model, training=False)
                model = self.xlr.prepare_model(model, evaluation_mode=True).to(
                    self.xlr.device
                )
            else:
                logger.debug(
                    "Inference loop: model already prepared (%s)",
                    model.__class__.__name__,
                )

            model = self.xlr.unwrap_model(model, keep_fp32_wrapper=False).eval()
        else:
            logger.debug(
                "Inference loop: no model provided - pre-existing evaluation files required"
            )

        # Output memory
        if path is None:
            results_remove_on_exit = True
            path = file_io.Path(
                f"//scratch/{self._params.project_name}/{str(self.session_id)}/results/{prefix}"  # .h5"
            )
            path.mkdir(parents=True, exist_ok=True)
        else:
            results_remove_on_exit = False
        try:
            path_memory = path / "memory"
            path_memory.mkdir(parents=True, exist_ok=True)
            path_evaluators = path / "evaluators"
            path_evaluators.mkdir(parents=True, exist_ok=True)

            results_mem = ResultsWriter(
                str(path_memory),
                samples_total,
                write_offset=batch_offsets[get_process_index()] * batch_size,
            )
            results_recovered = results_mem.recover()

            t_start_all = time.time()  # global start time
            samples_processed = 0
            if not results_recovered:
                if model is None:
                    msg = f"No model provided and no pre-existing evaluation files exist at the provided path {str(path)}"
                    raise RuntimeError(msg)
                logger.info(
                    "Inference loop: running on %d processes", get_process_count()
                )
                self._edge(
                    Event.ON_INFERENCE_BEGIN, loader=dataloader, path=path, model=model
                )

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    timings = ProfileAccumulator()
                    for sources, inputs in dataloader:
                        with profile(timings, "copy"):
                            inputs = inputs.to(self.xlr.device, non_blocking=True)
                        with profile(timings, "model"):
                            outputs = self.run_inference_step(model, inputs, optimizer)
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
                            for evaluator in handlers.values():
                                if not evaluator.enabled:
                                    continue
                                evaluator.update(
                                    results_merged,
                                    inputs,
                                    outputs,
                                    path=path_evaluators,
                                    sources=sources,
                                )
                        with profile(timings, "write"):
                            results_mem.add(results_merged, executor)
                            samples_processed += samples_in_batch
                        with profile(timings, "event"):
                            self._edge(
                                Event.ON_INFERENCE_STEP,
                                loader=dataloader,
                                inputs=inputs,
                                outputs=outputs,
                                sources=sources,
                                path=path,
                            )

                timings_summary = timings.to_summary()
                logger.info(
                    "Inference loop: timing report:\n%s",
                    create_table(timings_summary.reset_index(), format="wide"),
                )
                # HACK: Sleep for 1 second appears to help with memory consistency
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
                    path=path,
                )
            else:
                logger.info(
                    "Inference loop: recovered evaluation results from previous run"
                )

            # Compute metrics
            metrics: dict[str, T.Any] = {}
            if not results_recovered:
                metrics.update(
                    _build_speed_metrics(
                        prefix,
                        t_start_all,
                        sample_amount=samples_processed,
                        step_amount=math.ceil(samples_processed * get_process_count()),
                    )
                )
            visuals: dict[str, pil_image.Image | plt.Figure] = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                storage = results_mem.read(executor)
                for key, evaluator in handlers.items():
                    if not evaluator.enabled:
                        continue
                    logger.debug("Evaluating %s: %s", key, repr(evaluator))
                    metrics.update(
                        evaluator.compute(
                            storage, device=self.xlr.device, path=path_evaluators
                        )
                        or {}
                    )
                    if check_main_process():
                        visuals.update(
                            evaluator.render(storage, path=path_evaluators) or {}
                        )
            self._store_visualizations(visuals, prefix=prefix)

            barrier()
        finally:
            results_mem.close()
            if self.xlr.is_main_process and results_remove_on_exit:
                logger.info("Inference loop: cleaning up %s", path)
                shutil.rmtree(path, ignore_errors=True)
            else:
                logger.info("Inference loop: results stored at %s", path)
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
        if EngineStatus.EXPERIMENT_TRACKERS_STARTED in self.status:
            if not restart:
                logger.debug("Start trackers: skipping (already started)")
                return
            logger.debug("Start trackers: performing restart")
            self._stop_experiment_trackers()

        barrier()

        logger.info("Start trackers: initializing")

        self.status |= EngineStatus.EXPERIMENT_TRACKERS_STARTED
        self._state.step_experiment = 0
        self.xlr.trackers.clear()

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
                "name": " ".join(
                    [self.config_name, *self.config.get("CLI_OVERRIDES", [])]
                ),
                "job_type": job_type,
                "reinit": True,
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

        barrier()

    def _stop_experiment_trackers(self) -> None:
        """
        Stop the experiment trackers. Run has been finished and cannot be logged to
        anymore.
        """
        if EngineStatus.EXPERIMENT_TRACKERS_STARTED in self.status:
            logger.info("Stop trackers: marking experiment as finished")
            for tracker in self.xlr.trackers:
                tracker.finish()
            self.xlr.trackers.clear()
            self.status &= ~EngineStatus.EXPERIMENT_TRACKERS_STARTED
        else:
            logger.info("Stop trackers: skipping (not started)")

    def _train_handle_signals(
        self,
        tr_loss: dict[str, Tensor] | None,
        model: nn.Module,
        optimizer: AcceleratedOptimizer,
        timings_accumulator: ProfileAccumulator | None,
        *,
        trial: Trial | None,
    ) -> None:
        """
        Called at the end of every step and epoch to log, save, and evaluate the model.
        Steps could be skipped depending on the configuration.
        """

        # SIGNAL: logging
        if self._signal.should_log and not self._logged_in_last_step:
            assert tr_loss is not None

            self.status &= ~EngineStatus.FINDING_BATCH_SIZE
            steps_passed = self._state.step - self._step_last_logged

            # Allocate a log to store entries that will be logged to the experiment
            logs: dict[str, float] = {}
            logs["optimizer/lr"] = _get_learning_rate(optimizer)

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = {
                loss_key: reduce(loss_item, mode="mean", inplace=False)
                for loss_key, loss_item in tr_loss.items()
            }
            for k, f in tr_loss_scalar.items():
                v = (f.wait() / steps_passed).item()
                tr_loss[k].zero_()
                logs["losses/" + k] = round(v, 4)

            # Store FLOPs
            # self.store_flops()

            # Timings
            if timings_accumulator is not None:
                if timings_accumulator.steps_recorded > 0:
                    timings_summary = timings_accumulator.to_summary()
                    for k, v in timings_summary["mean"].to_dict().items():
                        logs["engine/time_train_" + k] = v
                timings_accumulator.reset()
            # logger.info("Timing report:\n%s", create_table(timings_summary.reset_index(), format="wide"))

            # Push logs
            self._step_last_logged = self._state.step
            self._push_logs(logs)

        # SIGNAL: save model
        if self._signal.should_save and not self._saved_in_last_step:
            logger.info(
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
            logger.info(
                "Starting evaluation cycle @ step %d / epoch %d",
                self._state.step,
                self._state.epoch,
            )

            self.run_evaluation(
                lambda *args, **kwargs: model, trial=trial, optimizer=optimizer
            )
            self._step_last_evaluated = self._state.step

    def _push_logs(self, logs: dict[str, T.Any]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Parameters
        ----------
        logs : dict[str, float]
            The logs to be logged.
        """
        logs["engine/epoch"] = round(self._state.epoch, 6)
        logs["engine/step"] = self._state.step
        logs["engine/gradient_accumulation"] = self._state.gradient_accumulation
        logs["engine/stage"] = self._state.stage
        logs["engine/epoch_step"] = self.xlr.step
        logs["engine/status"] = self.status
        logs["engine/step_experiment"] = self._state.step_experiment
        self._edge(Event.ON_LOG, logs=logs)  # NOTE: logs may be updated in-place
        self._state.register_logs(logs, max_history=self._params.logging_history)

        self.xlr.log(logs)

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

        logger.debug("Load weights: loading from %s", path)
        return load_checkpoint_and_dispatch(model, file_io.get_local_path(path))

    def _save_weights(
        self,
        path: Pathable | None,
        model: nn.Module,
        *,
        max_shard_size: int = 10_000_000_000,
    ) -> str:
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
                max_shard_size=max_shard_size,
            )
            _cleanup_generated_items(
                self.models_dir, self._params.save_total_limit or 1
            )

        barrier()

        return str(path)

    def _load_state(self, path: Pathable | None) -> None:
        """
        Load the engine state from the given path, if no path is given, the last checkpoint is used.
        """
        if path is not None:
            path = file_io.Path(path)
        elif self._recover_path is not None:
            path = file_io.Path(self._recover_path)
        else:
            if (
                not self.states_dir.is_dir()
                or len(list(self.states_dir.iterdir())) == 0
            ):
                msg = "No engine state path given and no automatic checkpoints found."
                raise FileNotFoundError(msg)
            path = _find_recent_generated_item(self.states_dir)
        self.xlr.load_state(str(path))

        logger.info(
            "Loaded state: %s\n%s", path, create_table(self._state.state_dict())
        )

    def _save_state(self, path: Pathable | None) -> str:
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
        self, visuals: dict[str, pil_image.Image | plt.Figure], prefix: str
    ) -> None:
        """
        Store visualizations that are provided as a mapping of (key) -> (PIL image).
        """

        import wandb

        if not check_main_process():
            return

        logger.info(
            f"Storing visualizations ({len(visuals)} total): {list(visuals.keys())}"
        )

        for key, img in visuals.items():
            if self._params.eval_write_visuals:
                img_path = (
                    file_io.Path(self.xlr.project_dir)
                    / "visuals"
                    / f"{prefix}-{self._state.step}"
                    / f"{key}"
                )
                img_path.parent.mkdir(parents=True, exist_ok=True)

                if isinstance(img, plt.Figure):
                    img.savefig(img_path.with_suffix(".eps"))
                elif isinstance(img, pil_image.Image):
                    img.save(img_path.with_suffix(".png"))
                else:
                    logger.warning(
                        "Visualizations: cannot save image type %s", type(img)
                    )
            if wandb.run is not None:
                if isinstance(img, pil_image.Image):
                    img_wandb = wandb.Image(img)
                else:
                    img_wandb = img
                try:
                    wandb.log({f"{prefix}/{key}": img_wandb}, commit=False)
                except Exception as err:
                    logger.warning("WandB: failed to log image %s: %s", key, err)

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


@retry_if_cuda_oom(cpu_ok=False)
def _forward(model: nn.Module, args: tuple[T.Any, ...]) -> ModelOutput:
    torch.compiler.cudagraph_mark_step_begin()
    return model(*args)


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


def _find_recent_generated_item(path: Pathable) -> str | None:
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
    return _read_session_name()


def _generate_experiment_id() -> str:
    """
    Generate a unique ID for the experiment.
    """
    import wandb.util

    return str(wandb.util.generate_id(length=8))
