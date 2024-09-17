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
from accelerate.utils import DynamoBackend

from unipercept import file_io
from unipercept.log import create_table, logger

if T.TYPE_CHECKING:
    from unipercept.engine import EngineParams

__all__ = ["Accelerator", "find_executable_batch_size", "StatefulObject"]


class StatefulObject(T.Protocol):
    """
    Protocol for classes that have a ``state_dict()`` and ``load_state_dict()`` method.
    """

    def state_dict(self) -> dict[str, T.Any]: ...

    def load_state_dict(self, state_dict: dict[str, T.Any]) -> None: ...


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
            InitProcessGroupKwargs,
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
                    **params.ddp_kwargs,
                ),
                InitProcessGroupKwargs(
                    **params.process_group_kwargs,
                ),
            ],
            step_scheduler_with_optimizer=False,
            log_with=list(params.trackers),
            dataloader_config=DataLoaderConfiguration(
                dispatch_batches=None, split_batches=False, non_blocking=True
            ),
            gradient_accumulation_steps=1,
            device_placement=True,
            **kwargs,
            dynamo_backend=DynamoBackend.NO,  # Set to 'NO' as we will configure it manually
        )
        acc._dynamo_from_params(params)

        return acc

    def _dynamo_from_params(self, params: EngineParams) -> None:
        """
        Configure the dynamo plugin from the engine parameters.

        Any values initialized in the Accelerate environment variables will be overridden
        by the values in the engine parameters. This ensures that each model instance and
        run can be traced back to a single configuration, without having to store the
        system environment variables.
        """
        from unipercept.config.env import get_env

        plugin: TorchDynamoPlugin | None = self.state.dynamo_plugin
        if plugin is None:
            msg = "Dynamo plugin is not initialized!"
            raise RuntimeError(msg)

        # Check if the backend is valid
        backend = params.compiler_backend
        if backend is not None:
            backend = backend.strip().upper()
            try:
                backend = DynamoBackend(backend)
            except ValueError as e:
                msg = f"Invalid compiler backend '{backend}', available: {list(DynamoBackend)}"
                raise ValueError(msg) from e
        else:
            backend = DynamoBackend.NO

        # Override the Accelerate 'plugin' settings
        plugin.backend = backend
        plugin.mode = params.compiler_mode
        plugin.fullgraph = params.compiler_fullgraph
        plugin.disable = get_env(bool, "UP_ENGINE_COMPILE_DISABLE", default=False)
        plugin.options = params.compiler_options
        plugin.dynamic = params.compiler_dynamic

        # Reset if configured
        torch._dynamo.config.suppress_errors = params.compiler_suppress_errors
        torch._dynamo.config.optimize_ddp = params.compiler_optimize_ddp

        # Debugging
        config = plugin.to_dict()
        config["suppress_errors"] = torch._dynamo.config.suppress_errors
        config["optimize_ddp"] = torch._dynamo.config.optimize_ddp

        logger.debug("Compiler configuration\n%s", create_table(config, format="long"))

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
