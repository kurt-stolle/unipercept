"""
Defines the interface for a perception model.
"""

from __future__ import annotations

import abc
import copy
import typing as T

import torch
import typing_extensions as TX
from tensordict import LazyStackedTensorDict, TensorDictBase
from torch import Tensor, nn
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

from unipercept.log import logger
from unipercept.types import Pathable

from ._io import InputData

__all__ = [
    "ModelInput",
    "ModelAdapter",
    "ModelBase",
    "ModelFactory",
    "ModelOutput",
]


#########################
# BASE CLASS FOR MODELS #
#########################

ModelInput = InputData | TensorDictBase | dict[str, Tensor]


class ModelOutput(T.NamedTuple):
    """
    The output of a model. This is a dictionary that can contain any number of tensors, but must at least contain
    a key "pred" that contains the model's prediction.
    """

    losses: dict[str, Tensor] | None
    predictions: list[TensorDictBase] | LazyStackedTensorDict | TensorDictBase | None


class ModelBase(nn.Module):
    """
    Defines the interface for a perception model. Defines the interface used throughout `unipercept`.

    Notes
    -----
    This class is abstract and cannot be instantiated directly. Instead, use :class:`ModelFactory` to instantiate a
    model from a configuration.

    Additionally, while this package defines a structured input data format, models are free to define their
    own, the interface only requires that the inptus and outputs are instances of a :class:`tensordict.TensorDictBase`
    subclass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abc.abstractmethod
    def select_inputs(self, data: ModelInput, **kwargs) -> tuple[T.Any, ...]:
        msg = f"Method `select_inputs` must be implemented for {cls.__name__}"
        raise NotImplementedError(msg)

    def predict(
        self, data: ModelInput, **kwargs
    ) -> list[TensorDictBase] | LazyStackedTensorDict | TensorDictBase | None:
        self = self.eval()
        with torch.inference_mode():
            inputs = self.select_inputs(data, **kwargs)
            return self(*inputs).predictions

    def losses(self, data: ModelInput, **kwargs) -> dict[str, Tensor]:
        self = self.train()
        inputs = self.select_inputs(data, **kwargs)
        return self(*inputs).losses

    if T.TYPE_CHECKING:

        def __call__(self, *args: Tensor) -> ModelOutput: ...


class ModelFactory:
    def __init__(
        self,
        model_config,
        weights: Pathable | None = None,
        freeze_weights: bool = False,
        compile: bool | dict[str, T.Any] = False,
    ):
        self.model_config = model_config
        self.weights = weights or None
        self.freeze_weights = freeze_weights

        if isinstance(compile, bool):
            self.compile = {} if compile else None
        else:
            self.compile = compile

    def __call__(
        self,
        *,
        weights: Pathable | None = None,
        overrides: T.Sequence[str] | T.Mapping[str, T.Any] | None = None,
    ) -> ModelBase:
        """
        TODO interface not clearly defined yet
        """
        from unipercept import load_checkpoint, read_checkpoint
        from unipercept.config.lazy import apply_overrides, instantiate

        # Configuration
        model_config = copy.deepcopy(self.model_config)
        if overrides is not None:
            if isinstance(overrides, T.Mapping):
                overrides_list = [f"{k}={v}" for k, v in overrides.items()]
            else:
                overrides_list = list(overrides)
            logger.info(
                "Model factory: config %s",
                ", ".join(overrides_list) if len(overrides_list) > 0 else "(none)",
            )
            model_config = apply_overrides(model_config, overrides_list)
        else:
            logger.info("Model factory: config has no overrides")

        # Instantiate model
        model = T.cast(ModelBase, instantiate(self.model_config))

        # Compile if options (kwargs to torch.compile) are set
        if self.compile is not None:
            logger.info("Model factory: compile options: %s", self.compile)
            model = torch.compile(model, **self.compile)
        else:
            logger.info("Model factory: compile disabled")

        # Load weights
        if weights is None:
            weights = self.weights
        if weights is not None:
            logger.info("Model factory: using weights from %s", weights)
            load_checkpoint(weights, model)
        else:
            logger.info("Model factory: using random initialization")

        # Freeze weights from the **initialization** checkpoint if requested
        if self.freeze_weights:
            freeze_keys = read_checkpoint(self.weights).keys()
            counter = 0
            for name, param in model.named_parameters():
                if name in freeze_keys:
                    param.requires_grad = False
                    logger.debug("Freezing parameter: %s (imported)", name)
                    counter += 1
            logger.info("Model factory: frozen %d parameters (imported)", counter)

        return model


class ModelAdapter(nn.Module):
    """
    A model may take rich input/output format (e.g. dict or custom classes),
    but `torch.jit.trace` requires tuple of tensors as input/output.
    This adapter flattens input/output format of a model so it becomes traceable.

    Notes
    -----
    This implementation is based on the Detectron2 ``TracingAdapter`` class.
    We use a custom implementation because Detectron2's implementation is not
    compatible with PyTree-based flattening.
    """

    flattened_inputs: tuple[Tensor, ...]
    inputs_schema: TreeSpec | None
    outputs_schema: TreeSpec | None

    def __init__(
        self,
        model: nn.Module,
        inputs,
        inference_func: T.Callable | None = None,
        allow_non_tensor: bool = False,
    ):
        """
        Parameters
        ----------

        model
            An nn.Module
        inputs
            An input argument or a tuple of input arguments used to call model.
            After flattening, it has to only consist of tensors.
        inference_func
            A callable that takes (model, *inputs), calls the
            model with inputs, and return outputs. By default it
            is ``lambda model, *inputs: model(*inputs)``. Can be override
            if you need to call the model differently.
        allow_non_tensor
            Allow inputs/outputs to contain non-tensor objects.
            This option will filter out non-tensor objects to make the
            model traceable, but ``inputs_schema``/``outputs_schema`` cannot be
            used anymore because inputs/outputs cannot be rebuilt from pure tensors.
            This is useful when you're only interested in the single trace of
            execution (e.g. for flop count), but not interested in
            generalizing the traced graph to new inputs.
        """

        super().__init__()

        if isinstance(
            model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
        ):
            model = model.module
        self.model = model
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        self.inputs = inputs
        self.allow_non_tensor = allow_non_tensor
        if inference_func is None:
            inference_func = lambda model, *inputs: model(*inputs)  # noqa
        self.inference_func = inference_func
        inputs_flat, inputs_spec = tree_flatten(inputs)

        if not all(isinstance(x, Tensor) for x in inputs_flat):
            if self.allow_non_tensor:
                inputs_flat = [x for x in inputs_flat if isinstance(x, Tensor)]
                inputs_spec = None
            else:
                for input in inputs_flat:
                    if isinstance(input, Tensor) or input is None:
                        continue
                    raise ValueError(
                        "Inputs for tracing must only contain tensors. "
                        f"Got a {type(input)} instead."
                    )

        self.flattened_inputs = tuple(inputs_flat)  # type: ignore
        self.inputs_schema = inputs_spec
        self.outputs_schema = None

    @TX.override
    def forward(self, *args: Tensor):
        with torch.no_grad():
            if self.inputs_schema is not None:
                inputs_orig_format = tree_unflatten(list(args), self.inputs_schema)
            else:
                if args != self.flattened_inputs:
                    msg = (
                        "TracingAdapter does not contain valid inputs_schema."
                        " So it cannot generalize to other inputs and must be"
                        " traced with `.flattened_inputs`."
                    )
                    raise ValueError(msg)
                inputs_orig_format = self.inputs

            outputs = self.inference_func(self.model, *inputs_orig_format)
            flattened_outputs, schema = tree_flatten(outputs)

            flattened_output_tensors = tuple(
                [x for x in flattened_outputs if isinstance(x, Tensor)]
            )
            if len(flattened_output_tensors) < len(flattened_outputs):
                if self.allow_non_tensor:
                    flattened_outputs = flattened_output_tensors
                    self.outputs_schema = None
                else:
                    raise ValueError(
                        "Model cannot be traced because some model outputs "
                        "cannot flatten to tensors."
                    )
            elif self.outputs_schema is None:
                self.outputs_schema = schema
            else:
                assert self.outputs_schema == schema, (
                    "Model should always return outputs with the same "
                    "structure so it can be traced!"
                )
            return flattened_outputs

    def _create_wrapper(self, traced_model):
        """
        Return a function that has an input/output interface the same as the
        original model, but it calls the given traced model under the hood.
        """

        def forward(*args):
            assert self.outputs_schema is not None

            flattened_inputs, _ = tree_flatten(args)
            flattened_outputs = traced_model(*flattened_inputs)
            return tree_unflatten(flattened_outputs, self.outputs_schema)

        return forward
