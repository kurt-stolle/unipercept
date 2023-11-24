"""
Implements a mapper that applies transformations to the input data,
where our version of the mapper especially ensures that geometric augmentations are
applied consistently to all frames of a paired record.

We use the term `ops` instead of `transforms` because the latter is easily confused with
a camera transform, e.g. movement of an observer.
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
import torchvision.transforms.v2 as tvt2
import torchvision.ops
from torchvision import disable_beta_transforms_warning as __disable_warning
from typing_extensions import override
from unicore.utils.pickle import as_picklable

from .tensors import BoundingBoxes, PanopticMap, DepthMap, BoundingBoxFormat
from unipercept.utils.logutils import get_logger

if T.TYPE_CHECKING:
    from unipercept.model import InputData

_logger = get_logger(name=__file__)


__disable_warning()


__all__ = ["apply_dataset", "Op", "CloneOp", "NoOp", "TorchvisionOp"]


########################################################################################################################
# BASE CLASS FOR OPS
########################################################################################################################


class Op(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for input operations. All operations are applied in-place.
    """

    def __init__(self) -> None:
        super().__init__()

    @override
    def forward(self, inputs: InputData) -> InputData:
        assert len(inputs.batch_size) == 0, f"Expected a single batched data point, got {inputs.batch_size}!"
        inputs = self._run(inputs)
        return inputs

    @abc.abstractmethod
    def _run(self, inputs: InputData) -> InputData:
        raise NotImplementedError(f"{self.__class__.__name__} is missing required implemention!")

    if T.TYPE_CHECKING:

        @override
        def __call__(self, inputs: InputData) -> InputData:
            ...


########################################################################################################################
# BASIC OPS
########################################################################################################################


class NoOp(Op):
    """Do nothing."""

    @override
    def _run(self, inputs: InputData) -> InputData:
        return inputs


class PinOp(Op):
    """Pin the input data to the device."""

    @override
    def _run(self, inputs: InputData) -> InputData:
        inputs = inputs.pin_memory()
        return inputs


class LogOp(NoOp):
    """Log the input data."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.register_forward_hook(self._log)  # type: ignore

    @staticmethod
    def _log(mod, inputs: InputData, outputs: tuple[list[str], InputData]) -> None:
        ids_str = ", ".join(inputs.ids)
        print(f"Applying ops on: '{ids_str}'...")


class CloneOp(Op):
    """Copy the input data."""

    @override
    def _run(self, inputs: InputData) -> InputData:
        inputs = inputs.clone(recurse=True)
        return inputs


########################################################################################################################
# TORCHVISION: Wrappers for torchvision transforms
########################################################################################################################


class TorchvisionOp(Op):
    """Wrap transforms from the torchvision library as an Op."""

    def __init__(self, transforms: T.Sequence[tvt2.Transform] | tvt2.Transform, *, verbose=False) -> None:
        super().__init__()

        self._verbose = verbose

        if isinstance(transforms, tvt2.Compose):
            self._transforms = transforms
            warnings.warn("Expected transforms to be a sequence or transform, got a `Compose`!", stacklevel=2)
        elif isinstance(transforms, T.Sequence):
            self._transforms = tvt2.Compose(transforms)
        elif isinstance(transforms, tvt2.Transform):
            self._transforms = tvt2.Compose([transforms])
        else:
            raise ValueError(f"Expected transforms to be a sequence or transform`, got {transforms}!")

    @override
    def _run(self, inputs: InputData) -> InputData:
        from .tensors.registry import pixel_maps

        if inputs.motions is not None:
            raise NotImplementedError("Transforms for motion data not supported!")

        inputs.captures = self._transforms(inputs.captures.fix_subtypes_())

        # caps_tf = []
        # for item in inputs.captures:
        #     for key, value in item.items():
        #         # Skip non-pixel maps
        #         if not isinstance(value, tuple(pixel_maps)):
        #             continue
        #         # Apply transforms
        #         value_tf = self._transforms(value)
        #         # value_tf.squeeze_(0)

        #         if self._verbose:
        #             _logger.debug(f"Transformed {key=} from a tensor {tuple(value.shape)} to {tuple(value_tf.shape)}")

        #         setattr(item, key, value_tf)
        #     caps_tf.append(item)

        # inputs.captures = torch.stack(caps_tf)

        return inputs


########################################################################################################################
# PSEUDO MOTION
########################################################################################################################


class PseudoMotion(Op):
    def __init__(
        self, frames: int, size: int | T.Sequence[int, int] = 512, scale=1.33, rotation=5, shear=1, p_reverse=0.5
    ):
        super().__init__()

        if scale < 1:
            raise ValueError(f"{scale=}")

        if frames <= 0:
            raise ValueError(f"{frames=}")
        elif frames == 1:
            warnings.warn(f"No pseudo motion is added when {frames=}", stacklevel=2)

        if isinstance(size, int):
            size_crop = (size, size)
        else:
            size_crop = size

        self._p_reverse = p_reverse
        self._out_frames = frames
        self._select = TorchvisionOp([tvt2.RandomCrop(size=size_crop)])
        self._upscale = TorchvisionOp(
            [
                tvt2.Resize(tuple(int(s * scale) for s in size_crop), antialias=True),
                tvt2.RandomAdjustSharpness(1.5),
                tvt2.RandomAffine(shear=(-shear, shear), degrees=(-rotation, rotation)),
                tvt2.RandomPhotometricDistort(),
                tvt2.GaussianBlur((5, 9)),
            ]
        )

    @override
    def _run(self, inputs: InputData) -> InputData:
        assert len(inputs.batch_size) == 0

        bs = list(inputs.captures.batch_size)
        assert bs[-1] == 1, f"Data already is a sequence: {inputs.captures.batch_size}"

        inp_list: list[InputData] = []

        for i in range(self._out_frames):
            inp_prev = inputs if i == 0 else self._upscale(inp_list[i - 1].clone())
            inp_next = self._select(inp_prev)
            inp_list.append(inp_next)

        reverse = torch.rand(1).item() < self._p_reverse

        if inputs.captures is not None:
            caps = [item.captures for item in inp_list]
            if reverse:
                caps.reverse()
            inputs.captures = torch.cat(caps, dim=0)
        if inputs.motions is not None:
            mots = [item.motions for item in inp_list]
            if reverse:
                mots.reverse()
            inputs.motions = torch.cat(mots, dim=0)

        return inputs


########################################################################################################################
# BOXES FROM MASKS
########################################################################################################################


class BoxesFromMasks(Op):
    """
    Adds bounding boxes for each ground truth mask in the input segmentation.
    """

    def __init__(self):
        super().__init__()

    @override
    def _run(self, inputs: InputData) -> InputData:
        assert len(inputs.batch_size) == 0

        caps = inputs.captures.fix_subtypes_()
        if caps.segmentations is not None:
            boxes = []
            for cap in caps:
                segs = torch.stack([m for _, m in cap.segmentations.as_subclass(PanopticMap).get_instance_masks()])
                boxes.append(torchvision.ops.masks_to_boxes(segs))

            h, w = inputs.captures.images.shape[-2:]
            inputs.captures.boxes = [BoundingBoxes(b, format=BoundingBoxFormat.XYXY, canvas_size=(h, w)) for b in boxes]

        return inputs


########################################################################################################################
# TRANSFORMED DATASETS
########################################################################################################################

_D = T.TypeVar("_D", bound=torch_data.Dataset, contravariant=True)


class _TransformedIterable(torch_data.IterableDataset["InputData"], T.Generic[_D]):
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
    def __iter__(self) -> T.Iterator[InputData]:
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


class _TransformedMap(torch_data.Dataset["InputData"], T.Generic[_D]):
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
    def __getitem__(self, idx: int | str) -> tuple[InputData]:
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
