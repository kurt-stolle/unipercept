"""
Implements a mapper that applies transformations to the input data,
where our version of the mapper especially ensures that geometric augmentations are
applied consistently to all frames of a paired record.
"""
from __future__ import annotations

import abc
import random
import typing as T
import warnings

import torch
import torch.nn
import torch.types
import torch.utils.data as torch_data
import unipercept.data.points as data_points
from torchvision import disable_beta_transforms_warning as __disable_warning
from typing_extensions import override
from unicore.utils.pickle import as_picklable

from .sets import PerceptionDataset

__disable_warning()


__all__ = ["apply_dataset", "Op", "CloneOp", "NoOp", "TorchvisionOp"]


# ---------------------- #
# Baseclass and protocol #
# ---------------------- #


class Op(torch.nn.Module, metaclass=abc.ABCMeta):
    """Base class for input operations. All operations are applied in-place."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def forward(self, inputs: data_points.InputData) -> data_points.InputData:
        assert len(inputs.batch_size) == 0, f"Expected a single batched data point, got {inputs.batch_size}!"
        inputs = self._run(inputs)
        return inputs

    @abc.abstractmethod
    def _run(self, inputs: data_points.InputData) -> data_points.InputData:
        raise NotImplementedError(f"{self.__class__.__name__} is missing required implemention!")

    if T.TYPE_CHECKING:

        @override
        def __call__(self, inputs: data_points.InputData) -> tuple[list[str], data_points.InputData]:
            ...


# ------------------------------ #
# Basic and primitive operations #
# ------------------------------ #


class NoOp(Op):
    """Do nothing."""

    @override
    def _run(self, inputs: data_points.InputData) -> data_points.InputData:
        return inputs


class PinOp(Op):
    """Pin the input data to the device."""

    @override
    def _run(self, inputs: data_points.InputData) -> data_points.InputData:
        inputs = inputs.pin_memory()
        return inputs


class LogOp(NoOp):
    """Log the input data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_forward_hook(self._log)  # type: ignore

    @staticmethod
    def _log(mod, inputs: data_points.InputData, outputs: tuple[list[str], data_points.InputData]) -> None:
        ids_str = ", ".join(inputs.ids)
        print(f"Applying ops on: '{ids_str}'...")


class CloneOp(Op):
    """Copy the input data."""

    @override
    def _run(self, inputs: data_points.InputData) -> data_points.InputData:
        inputs = inputs.clone(recurse=True)
        return inputs


# ---------------------------------- #
# Torchvision transforms as Ops #
# ---------------------------------- #

import torchvision.transforms.v2 as tv_transforms


class TorchvisionOp(Op):
    """Wrap transforms from the torchvision library as an Op."""

    def __init__(
        self,
        transforms: T.Sequence[tv_transforms.Transform] | tv_transforms.Transform,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(transforms, tv_transforms.Compose):
            self.transforms = transforms
            warnings.warn("Expected transforms to be a sequence or transform, got a `Compose`!")
        elif isinstance(transforms, T.Sequence):
            self.transforms = tv_transforms.Compose(transforms)
        elif isinstance(transforms, tv_transforms.Transform):
            self.transforms = tv_transforms.Compose([transforms])
        else:
            raise ValueError(f"Expected transforms to be a sequence or transform`, got {transforms}!")

    @override
    def _run(self, inputs: data_points.InputData) -> data_points.InputData:
        for item in inputs.captures.unsqueeze(0):
            for key, value in item.items():
                # Skip non-pixel maps
                if not isinstance(value, tuple(data_points.registry.pixel_maps)):
                    continue
                # Apply transforms
                value_tf = self.transforms(value)
                value_tf.squeeze_(0)
                setattr(item, key, value_tf)

        return inputs


# ------------------------------------------------- #
# Transformed versions of map and iterable datasets #
# ------------------------------------------------- #


_D = T.TypeVar("_D", bound=torch_data.Dataset, contravariant=True)


class _TransformedIterable(torch_data.IterableDataset[data_points.InputData], T.Generic[_D]):
    """Applies a sequence of transformations to an iterable dataset."""

    __slots__ = ("_set", "_fns")

    def __init__(self, dataset: _D, fns: T.Sequence[Op]):
        self._set = dataset
        self._fns = list(as_picklable(fn) for fn in fns)

        assert len(self) >= 0

    def __len__(self):
        if not isinstance(self._set, T.Sized):
            raise ValueError(f"Dataset {self._set} must be sized!")
        return len(self._set)

    def __getnewargs__(self) -> tuple[_D, list[Op]]:
        return self._set, self._fns

    @override
    def __str__(self):
        return f"{str(self._set)} ({len(self._fns)} transforms)"

    @override
    def __repr__(self):
        return f"<{repr(self._set)} x {len(self._fns)} transforms>"

    @override
    def __iter__(self) -> T.Iterator[data_points.InputData]:
        it = iter(self._set)
        while True:
            try:
                inputs = next(it)
            except StopIteration:
                return

            for fn in self._fns:
                inputs = fn(inputs)
                if inputs is None:
                    warnings.warn("Transformed data is None, skipping!", stacklevel=2)
                    break
            else:
                yield inputs


class _TransformedMap(torch_data.Dataset[data_points.InputData], T.Generic[_D]):
    """Applies a sequence of transformations to an iterable dataset."""

    __slots__ = ("_set", "_fns", "_retry", "_fallback_candidates")

    def __init__(self, dataset: _D, fns: T.Sequence[Op], *, max_retry: int = 3):
        self._set = dataset
        self._fns = list(as_picklable(fn) for fn in fns)

        assert len(self) >= 0
        self._retry = max_retry
        self._fallback_candidates: set[int | str] = set()
        self._random = random.Random(42)

    def __len__(self):
        if not isinstance(self._set, T.Sized):
            raise ValueError(f"Dataset {self._set} must be sized!")
        return len(self._set)

    def __getnewargs__(self) -> tuple[_D, list[Op]]:
        return self._set, self._fns

    @override
    def __str__(self):
        return f"{str(self._set)} ({len(self._fns)} transforms)"

    @override
    def __repr__(self):
        return f"<{repr(self._set)} x {len(self._fns)} transforms>"

    @override
    def __getitem__(self, idx: int | str) -> tuple[data_points.InputData]:
        for _ in range(self._retry):
            inputs = self._set[idx]
            assert len(inputs.batch_size) == 0, f"Expected a single batched data point, got {inputs.batch_size}!"
            for fn in self._fns:
                inputs = fn(inputs)
                if inputs is None:
                    warnings.warn("Transformed data is None, skipping!", stacklevel=2)
                    break
            else:
                self._fallback_candidates.add(idx)
                return inputs
            self._fallback_candidates.discard(idx)
            idx = self._random.sample(list(self._fallback_candidates), k=1)[0]

        raise RuntimeError(f"Failed to apply transforms after {self._retry} retries!")


def apply_dataset(dataset: _D, actions: T.Sequence[Op]) -> _TransformedMap[_D] | _TransformedIterable[_D]:
    """Map a function over the elements in a dataset."""
    if isinstance(dataset, torch_data.IterableDataset):
        return _TransformedIterable(dataset, actions)
    else:
        return _TransformedMap(dataset, actions)
