"""
Integration with the ``accelerate`` package.
"""

from __future__ import annotations

import typing as T

import accelerate
import accelerate.utils
import torch
import torch._dynamo
import torch._dynamo.config
import torch.nn
import torch.types
import torch.utils.data
import typing_extensions as TX
from accelerate.accelerator import TorchDynamoPlugin

from unipercept import file_io
from unipercept.log import logger

if T.TYPE_CHECKING:
    from unipercept.engine import EngineParams

__all__ = ["Accelerator", "find_executable_batch_size", "StatefulObject"]

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.optimize_ddp = False


class StatefulObject(T.Protocol):
    """
    Protocol for classes that have a ``state_dict()`` and ``load_state_dict()`` method.
    """

    def state_dict(self) -> T.Dict[str, T.Any]: ...

    def load_state_dict(self, state_dict: T.Dict[str, T.Any]) -> None: ...


class Accelerator(accelerate.Accelerator):
    """
    Subclass of ``accelerate.Accelerator`` that adds support for various ``unipercept`` specific features.
    """

    @classmethod
    def from_engine_params(cls, params: EngineParams, root: file_io.Path) -> T.Self:
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
        from accelerate.utils import (
            DataLoaderConfiguration,
            DistributedDataParallelKwargs,
        )

        project_dir = root / "outputs"
        logging_dir = root / "logs"
        project_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

        kwargs = {}

        if params.training_precision not in {None, "", "default"}:
            kwargs["mixed_precision"] = params.training_precision

        acc = cls(
            project_dir=project_dir,
            project_config=ProjectConfiguration(
                project_dir=str(project_dir),
                logging_dir=str(logging_dir),
                automatic_checkpoint_naming=False,
                total_limit=params.save_total_limit or 1,
            ),
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=params.find_unused_parameters,
                    broadcast_buffers=True,  # False,
                    gradient_as_bucket_view=True,
                    static_graph=params.static_graph,
                ),
            ],
            step_scheduler_with_optimizer=False,
            log_with=list(params.trackers),
            dataloader_config=DataLoaderConfiguration(
                dispatch_batches=None, split_batches=False
            ),
            gradient_accumulation_steps=1,
            device_placement=True,
            **kwargs,
            # dynamo_backend=None,
        )
        acc.dynamo = cls.dynamo_from_params(params)

        return acc

    @classmethod
    def dynamo_from_params(cls, params: EngineParams) -> TorchDynamoPlugin:
        from accelerate.utils import DynamoBackend, TorchDynamoPlugin

        from unipercept.config import get_env

        if get_env(bool, "UP_ENGINE_COMPILE_RESET", default=params.compiler_reset):
            torch._dynamo.reset()
        backend = get_env(
            str, "UP_ENGINE_COMPILE_BACKEND", default=params.compiler_backend
        )
        if backend is None:
            backend = "no"
        assert isinstance(backend, str), type(backend)
        backend = DynamoBackend(backend.upper())
        if backend == DynamoBackend.NO:
            logger.debug("Skipping moddel compliation. Disabled by user/parms.")
            return TorchDynamoPlugin()

        config = params.compiler_config
        config_mode = get_env(str, "UP_ENGINE_COMPILE_MODE", default=None)
        if config_mode is not None:
            config["mode"] = config_mode

        logger.debug("Using compiler backend '%s' with config %s", backend, str(config))

        return TorchDynamoPlugin(
            backend=DynamoBackend(backend),
            **config,
        )

    @TX.override
    def prepare_model(self, model: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        """
        Prepares the model for training. See ``accelerate.Accelerator.prepare_model`` for more information.
        """
        from unipercept.config import get_env

        prepared_model = super().prepare_model(model, *args, **kwargs)
        #     backend = get_env(str, "UP_ENGINE_COMPILE_BACKEND", default="inductor")
        #     if backend != "disabled":
        #         if get_env(bool, "UP_ENGINE_COMPILE_RESET", default=False):
        #             torch._dynamo.reset()
        #         logger.debug("Compiling model with backend '%s'.", backend)
        #         prepared_model.compile(backend=backend)
        #     else:
        #         logger.debug(
        #             "Got compile backend '%s'. Skipping model compilation.", backend
        #         )

        return prepared_model

    @TX.override
    def register_for_checkpointing(self, obj: StatefulObject) -> None:
        """
        Registers an object for checkpointing. See ``accelerate.Accelerator.register_for_checkpointing`` for more information.
        """
        return super().register_for_checkpointing(obj)


if T.TYPE_CHECKING:
    _P = T.ParamSpec("_P")
    _R = T.TypeVar("_R")

    _Fin: T.TypeAlias = T.Callable[T.Concatenate[int, _P], _R]
    _Fout: T.TypeAlias = T.Callable[_P, _R]

    @T.overload
    def find_executable_batch_size(
        function: _Fin[_P, _R],
        *,
        starting_batch_size: int = 128,
    ) -> _Fout[_P, _R]: ...

    @T.overload
    def find_executable_batch_size(
        function: None = None,
        *,
        starting_batch_size: int = 128,
    ) -> T.Callable[[_Fin[_P, _R]], _Fout[_P, _R]]: ...

    def find_executable_batch_size(
        function: _Fin | None = None,
        *,
        starting_batch_size: int = 128,
    ) -> T.Callable[[_Fin[_P, _R]], _Fout[_P, _R]] | _Fout[_P, _R]: ...

else:
    find_executable_batch_size = accelerate.utils.find_executable_batch_size
