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
from typing import override

import torch
import torch.nn
import torch.types
import torch.utils.data as torch_data
import torchvision.ops
import torchvision.transforms.v2 as tvt2
import torchvision.transforms.v2.functional
from torchvision import disable_beta_transforms_warning as __disable_warning

import unipercept.log as logger
import unipercept.model
from unipercept.config import env
from unipercept.data import tensors
from unipercept.utils.pickle import as_picklable

__disable_warning()

FILL_VALUES = {
    tensors.PanopticMap: tensors.PanopticMap.IGNORE,
    tensors.Image: 127,
}


def get_fill_values():
    """
    Get the fill values for the different types of tensors when padding.
    This function is useful for configuration files, where the keys may not be
    passed as a Python object.
    """
    return FILL_VALUES


########################################################################################
# BASE CLASS FOR OPS
########################################################################################


class OpReject(Exception):
    """
    Exception that is raised when a operation fails to modify the input data in a way
    that is acceptable.

    The current data point is discarded and the loader will continue with the next one.
    """

    def __init__(self, message: str):
        super().__init__(message)


class OpSkip(Exception):
    """
    Exception that is raised when a operation flags itself to be skipped.
    """

    def __init__(self, message: str):
        super().__init__(message)


class Op(torch.nn.Module):
    """
    Base class for input operations. All operations are applied in-place.
    """

    @staticmethod
    def transform_inputs(
        inputs, ops: T.Iterable[Op]
    ) -> T.Iterator[unipercept.model.InputData]:
        for fn in ops:
            try:
                inputs = fn(inputs)
            except OpSkip as e:
                if fn._verbose:
                    logger.debug("Operation was skipped: %s", e)
                continue
            except OpReject as e:
                if fn._verbose:
                    logger.warning("Operation rejected sample: %s", e, stacklevel=2)
                break
            if inputs is None:
                logger.warning("Transformed data is None, skipping!", stacklevel=2)
                break
            if not isinstance(inputs, unipercept.model.InputData):
                if isinstance(inputs, T.Sequence):
                    if fn._verbose:
                        logger.debug(
                            "Operation returned a sequence of %d items, splitting...",
                            len(inputs),
                        )
                    for item in inputs:
                        yield from Op.transform_inputs(item, ops)
                else:
                    msg = (
                        f"Expected an unipercept.model.InputData object, got {inputs}!"
                    )
                    raise ValueError(msg)
        else:
            yield inputs

    def __init__(self, *, verbose: bool | None = None, **kwargs) -> None:
        super().__init__(**kwargs)

        if verbose is None:
            verbose = env.get_env(bool, "UP_DATA_OPS_VERBOSE", default=False)

        self._verbose = verbose

    @override
    @torch.no_grad()
    def forward(
        self, inputs: unipercept.model.InputData
    ) -> unipercept.model.InputData | None:
        # assert len(inputs.batch_size) == 0, f"Expected a single batched data point, got {inputs.batch_size}!"
        if self._verbose:
            logger.debug("Running %s op...", self.__class__.__name__)
        return self._run(inputs)

    @abc.abstractmethod
    def _run(
        self, inputs: unipercept.model.InputData
    ) -> unipercept.model.InputData | None:
        msg = f"{self.__class__.__name__} is missing required implemention!"
        raise NotImplementedError(msg)

    if T.TYPE_CHECKING:

        @override
        def __call__(
            self, inputs: unipercept.model.InputData
        ) -> unipercept.model.InputData: ...


class CloneOp(Op):
    """Copy the input data."""

    @override
    def _run(self, inputs: unipercept.model.InputData) -> unipercept.model.InputData:
        return inputs.clone(recurse=True)


########################################################################################
# TORCHVISION: Wrappers for torchvision transforms
########################################################################################


class TorchvisionOp(Op):
    """Wrap transforms from the torchvision library as an Op."""

    def __init__(
        self, transforms: T.Sequence[tvt2.Transform] | tvt2.Transform, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(transforms, tvt2.Compose):
            self._transforms = transforms
        elif isinstance(transforms, T.Sequence):
            if len(transforms) == 0:
                self._transforms = None
            else:
                self._transforms = tvt2.Compose(transforms)
        elif isinstance(transforms, tvt2.Transform):
            self._transforms = transforms
        else:
            raise ValueError(
                f"Expected transforms to be a sequence or transform`, got {transforms}!"
            )

    @override
    def _run(self, inputs: unipercept.model.InputData) -> unipercept.model.InputData:
        if self._transforms is None:
            return inputs
        if inputs.motions is not None:
            raise NotImplementedError("Transforms for motion data not supported!")
        inputs.captures = self._transforms(inputs.captures.fix_subtypes_())

        return inputs


########################################################################################
# CROPPING
########################################################################################


class GuidedRandomCrop(Op):
    """
    Performs random cropping based on the (panoptic) segmentation map.
    """

    def __init__(
        self,
        size: T.Iterable[int] | tuple[int, int] | int,
        *,
        min_unique_classes: int = 2,
        min_unique_instances: int = 1,
        min_instance_area: float = 1e-2,
        min_valid_segmentation_area: float = 0.70,
        min_valid_depth_area: float = 0.10,
        max_iterations: int = 40,
        step_factor: int = 4,
        reject_on_failure: bool = False,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        size
            Output size of the crop. If a single integer is given, the output will be a square crop.
        verbose
            Whether to print debug information.

        """
        from unipercept.utils.function import to_2tuple

        assert min_unique_classes >= 0
        assert min_unique_instances >= 0
        assert min_instance_area >= 0
        assert min_valid_segmentation_area >= 0
        assert max_iterations >= 0

        super().__init__(**kwargs)

        self._size = to_2tuple(size)
        self._step_factor = step_factor
        self.min_unique_classes = min_unique_classes
        self.min_unique_instances = min_unique_instances
        self.min_instance_area = min_instance_area
        self.min_valid_segmentation_area = min_valid_segmentation_area
        self.min_valid_depth_area = min_valid_depth_area
        self.max_iterations = max_iterations
        self.reject_on_failure = reject_on_failure

    def _find_crop(
        self, panoptic: tensors.PanopticMap, depth: tensors.DepthMap | None
    ) -> tuple[int, int]:
        # TODO: what if only some conditions are met?

        assert (
            panoptic.ndim == 3
        ), f"Expected a panoptic map with shape PHW, got {panoptic.shape}!"

        # Only consider the first element in the sequence
        panoptic = panoptic[0, :, :].as_subclass(tensors.PanopticMap)

        # Compute step size
        step = min(panoptic.shape[-1], panoptic.shape[-2]) // min(
            self._size[-1], self._size[-2]
        )
        step *= self._step_factor

        # Height/width
        height, width = self._size
        top_choices = list(range(0, panoptic.shape[-2] - height + 1, step))
        left_choices = list(range(0, panoptic.shape[-1] - width + 1, step))

        # Compute total area and ignore area
        total_area = height * width

        # Create random crops until the conditions are met or the maximum number of iterations is reached
        for top, left in zip(
            random.choices(top_choices, k=self.max_iterations),
            random.choices(left_choices, k=self.max_iterations),
            strict=False,
        ):
            # Randomly select a crop
            crop = torchvision.transforms.v2.functional.crop_mask(
                panoptic, top, left, height, width
            ).as_subclass(tensors.PanopticMap)

            crop_sem = crop.get_semantic_map()
            crop_ins = crop.get_instance_map()
            crop_valid = crop_sem != panoptic.IGNORE

            # Compute the area of the ignore regions in the crop
            valid_area = (crop_valid).float().sum() / total_area
            if valid_area < self.min_valid_segmentation_area:
                continue

            # Compute the number of unique classes and instances in the crop
            unique_classes = torch.unique(crop_sem[crop_valid]).numel()
            if unique_classes < self.min_unique_classes:
                continue

            has_instances = crop_ins > 0

            unique_instances = torch.unique(crop_ins[has_instances]).numel()
            if unique_instances < self.min_unique_instances:
                continue

            # Compute the area of the instances in the crop
            instance_area = (has_instances).float().sum() / total_area

            if instance_area < self.min_instance_area:
                continue

            if depth is not None:
                crop_depth = torchvision.transforms.v2.functional.crop(
                    depth, top, left, height, width
                ).as_subclass(tensors.DepthMap)
                valid_depth_area = (crop_depth > 0).float().sum() / total_area
                if valid_depth_area < self.min_valid_depth_area:
                    continue

            return top, left
        msg = f"Failed to find a valid crop after {self.max_iterations} iterations!"
        if self._verbose:
            logger.warning(msg)
        if self.reject_on_failure:
            raise OpReject(msg)
        return random.choice(top_choices), random.choice(left_choices)

    @override
    def _run(self, inputs: unipercept.model.InputData) -> unipercept.model.InputData:
        from .tensors.registry import pixel_maps

        if inputs.motions is not None:
            raise NotImplementedError("Transforms for motion data not supported!")
        if inputs.captures.segmentations is None:
            raise ValueError("Expected a panoptic segmentation map!")

        top, left = self._find_crop(
            inputs.captures.segmentations, inputs.captures.depths
        )

        def apply_crop(x: torch.Tensor) -> torch.Tensor:
            if type(x) not in pixel_maps:
                return x
            return torchvision.transforms.v2.functional.crop(x, top, left, *self._size)

        inputs.captures = inputs.captures.fix_subtypes_().apply(
            apply_crop, batch_size=inputs.captures.batch_size
        )
        return inputs


class RequireValidContent(Op):
    """
    Removes samples that do not meet the specified criteria.
    """

    def __init__(self, valid_segmentation: bool = True, valid_depth: bool = True):
        super().__init__()
        self._valid_segmentation = valid_segmentation
        self._valid_depth = valid_depth

    @override
    def _run(
        self, inputs: unipercept.model.InputData
    ) -> unipercept.model.InputData | None:
        if self._valid_segmentation:
            if inputs.captures.segmentations is None:
                if self._verbose:
                    msg = f"Skipping {inputs.ids} because no segmentation is available!"
                    logger.warning(msg)
                return None
            if not (inputs.captures.segmentations != tensors.PanopticMap.IGNORE).any():
                if self._verbose:
                    msg = f"Skipping {inputs.ids} because all segments are ignored!"
                    logger.warning(msg)
                return None
        if self._valid_depth:
            if inputs.captures.depths is None:
                if self._verbose:
                    msg = f"Skipping {inputs.ids} because no depth map is available!"
                    logger.warning(msg)
                return None
            if not (inputs.captures.depths > 0).any():
                if self._verbose:
                    msg = f"Skipping {inputs.ids} because all depth is invalid!"
                    logger.warning(msg)
                return None
        return inputs


########################################################################################
# PSEUDO MOTION
########################################################################################


class PseudoMotion(Op):
    def __init__(
        self,
        frames: int,
        size: int | T.Sequence[int] = 512,
        scale=1.33,
        rotation=5,
        shear=1,
        p_reverse=0.5,
    ):
        super().__init__()

        if scale < 1:
            raise ValueError(f"{scale=}")

        if frames <= 0:
            raise ValueError(f"{frames=}")
        if frames == 1:
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
                tvt2.RandomAdjustSharpness(1.2),
                tvt2.RandomAffine(shear=(-shear, shear), degrees=(-rotation, rotation)),
            ]
        )

    @override
    def _run(self, inputs: unipercept.model.InputData) -> unipercept.model.InputData:
        if inputs.motions is not None:
            # TODO Implement pseudo motion for motion data (@kurt-stolle)
            raise NotImplementedError("Psuedo motion for motion data not supported!")

        if inputs.num_frames > self._out_frames:
            warnings.warn(
                f"Skipping pseudo motion for {inputs.ids} because {inputs.num_frames=} > {self._out_frames=}",
                stacklevel=2,
            )
            return inputs

        inp_list: list[unipercept.model.InputData] = []

        for i in range(self._out_frames):
            # Either select one of the input frames or upscale the previous pseudo-input
            if i < inputs.num_frames:
                inp_prev = inputs.extract_frame(i)
            else:
                inp_prev = self._upscale(inp_list[i - 1].clone())

            # Select a crop that matches the desired output size of the sequence
            inp_next = self._select(inp_prev)

            # Append to the list of frames
            inp_list.append(inp_next)

        # Random reversing of the sequence, whicih can be applied even when no pseudo-frames were generated (e.g. when the input sequence already has sufficient frames)
        reverse = torch.rand(1).item() < self._p_reverse

        if inputs.captures is not None:
            caps = [item.captures for item in inp_list]
            if reverse:
                caps.reverse()
            inputs.captures = torch.stack(caps, dim=0)
            # if reverse:
            #     mots.reverse()
            # inputs.motions = torch.cat(mots, dim=0)

        return inputs


########################################################################################
# COMPOSITION
########################################################################################


class PadToDivisible(Op):
    """
    Pads the input to be divisible by a given number.
    """

    def __init__(self, divisor: int):
        super().__init__()
        self._divisor = divisor

    @override
    def _run(self, inputs: unipercept.model.InputData) -> unipercept.model.InputData:
        from .tensors.registry import pixel_maps

        if inputs.motions is not None:
            raise NotImplementedError("Transforms for motion data not supported!")

        h, w = inputs.captures.images.shape[-2:]
        pad_h = (self._divisor - h % self._divisor) % self._divisor
        pad_w = (self._divisor - w % self._divisor) % self._divisor

        if pad_h == 0 and pad_w == 0:
            return inputs

        def apply_padding(x: torch.Tensor) -> torch.Tensor:
            if type(x) not in pixel_maps:
                return x

            return torchvision.transforms.v2.functional.pad(
                x,
                [0, 0, pad_w, pad_h],
                fill=next(
                    pad_value
                    for pad_value in (
                        FILL_VALUES.get(type(x)),
                        next(
                            (v for t, v in FILL_VALUES.items() if isinstance(x, t)),
                            None,
                        ),
                        0,
                    )
                    if pad_value is not None
                ),
            )

        inputs.captures = inputs.captures.fix_subtypes_().apply(
            apply_padding, batch_size=inputs.captures.batch_size
        )

        return inputs


########################################################################################
# BOXES FROM MASKS
########################################################################################


class BoxesFromMasks(Op):
    """
    Adds bounding boxes for each ground truth mask in the input segmentation.
    """

    def __init__(self):
        super().__init__()

    @override
    def _run(self, inputs: unipercept.model.InputData) -> unipercept.model.InputData:
        assert len(inputs.batch_size) == 0

        caps = inputs.captures.fix_subtypes_()
        if caps.segmentations is not None:
            boxes = []
            for cap in caps:
                segs = torch.stack(
                    [
                        m
                        for _, m in cap.segmentations.as_subclass(
                            tensors.PanopticMap
                        ).get_instance_masks()
                    ]
                )
                boxes.append(torchvision.ops.masks_to_boxes(segs))

            h, w = inputs.captures.images.shape[-2:]
            inputs.captures.boxes = [
                tensors.BoundingBoxes(
                    b, format=tensors.BoundingBoxFormat.XYXY, canvas_size=(h, w)
                )
                for b in boxes
            ]

        return inputs


########################################################################################
# TRANSFORMED DATASETS
########################################################################################

_D = T.TypeVar("_D", bound=torch_data.Dataset, contravariant=True)


class _TransformedIterable(
    torch_data.IterableDataset["unipercept.model.InputData"], T.Generic[_D]
):
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
    def __iter__(self) -> T.Iterator[unipercept.model.InputData]:
        it = iter(self._set)
        while True:
            try:
                item, data = next(it)
            except StopIteration:
                return
            for data in Op.transform_inputs(data, self._fns):
                yield item, data


class _TransformedMap[
    _Q,
    _R: "unipercept.model.InputData",
](torch_data.Dataset[tuple[_Q, _R]]):
    """Applies a sequence of transformations to an iterable dataset."""

    __slots__ = ("_set", "_fns", "_retry", "_fallback_candidates")

    def __init__(
        self,
        dataset: unipercept.data.sets.PerceptionDataset,
        fns: T.Sequence[Op],
        *,
        max_retry: int = 100,
    ):
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

    # def __getnewargs__(self) -> tuple[_D, list[Op]]:
    #    return self._set, self._fns

    @override
    def __str__(self):
        return f"{str(self._set)} ({len(self._fns)} transforms)"

    @override
    def __repr__(self):
        return f"<{repr(self._set)} x {len(self._fns)} transforms>"

    @override
    def __getitem__(self, idx: int | str) -> tuple[_Q, _R]:
        for _ in range(self._retry):
            item, data = self._set[idx]
            data = next(Op.transform_inputs(data, self._fns), None)
            if data is None:
                self._fallback_candidates.discard(idx)
                if len(self._fallback_candidates) == 0:
                    idx = self._random.randint(0, len(self) - 1)
                else:
                    idx = self._random.sample(list(self._fallback_candidates), k=1)[0]
            else:
                self._fallback_candidates.add(idx)
                return item, data

        raise RuntimeError(f"Failed to apply transforms after {self._retry} retries!")


def apply_dataset(
    dataset: _D, actions: T.Sequence[Op]
) -> _TransformedMap[_D] | _TransformedIterable[_D]:
    """Map a function over the elements in a dataset."""
    if isinstance(dataset, torch_data.IterableDataset):
        return _TransformedIterable(dataset, actions)
    return _TransformedMap(dataset, actions)
