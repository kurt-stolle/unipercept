"""
Integration with the ``accelerate`` package.
"""

from __future__ import annotations

import accelerate
import accelerate.utils

import typing as T
import torch
import torch.types
import torch.nn
import typing_extensions as TX

from unipercept import file_io

if T.TYPE_CHECKING:
    from unipercept.engine import EngineParams

__all__ = ["Accelerator"]


class Accelerator(accelerate.Accelerator):
    """
    Subclass of ``accelerate.Accelerator`` that adds support for various ``unipercept`` specific features.
    """

    @classmethod
    def from_engine_params(cls, params: EngineParams) -> T.Self:
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
            DistributedDataParallelKwargs,
            TorchDynamoPlugin,
        )

        root = file_io.Path(params.root).resolve()

        project_dir = root / "outputs"
        logging_dir = root / "logs"

        project_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

        acc = cls(
            project_dir=project_dir,
            project_config=ProjectConfiguration(
                project_dir=str(project_dir),
                logging_dir=str(logging_dir),
                automatic_checkpoint_naming=True,
                total_limit=4,
            ),
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=params.find_unused_parameters,
                    broadcast_buffers=True,  # False,
                    gradient_as_bucket_view=True,
                    static_graph=True,
                ),
            ],
            step_scheduler_with_optimizer=False,
            log_with=list(params.trackers),
            dispatch_batches=False,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            split_batches=False,
            device_placement=True,
            mixed_precision="bf16",
            dynamo_backend=None,
        )
        acc.state.dynamo_plugin = TorchDynamoPlugin(
            # backend=DynamoBackend.INDUCTOR,
        )
        return acc

    @TX.override
    def prepare_model(self, model: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        """
        Prepares the model for training. See ``accelerate.Accelerator.prepare_model`` for more information.
        """
        model = super().prepare_model(model, *args, **kwargs)

        return model
