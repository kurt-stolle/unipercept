"""
Defines the interface for a perception model.
"""

from __future__ import annotations

import abc
import typing as T
from dataclasses import field

import torch
import torch.nn as nn
from tensordict import LazyStackedTensorDict, TensorDict
from torchvision.tv_tensors import Image
from typing_extensions import override
from unicore.utils.tensorclass import Tensorclass

from unipercept.data.tensors import DepthMap, OpticalFlow, PanopticMap

__all__ = ["ModelBase", "InputData", "ModelOutput", "CaptureData", "MotionData"]


class CameraModel(Tensorclass):
    """
    Build pinhole camera calibration matrices.

    See: https://kornia.readthedocs.io/en/latest/geometry.camera.pinhole.html
    """

    image_size: torch.Tensor  # shape: ..., 2 (height, width)
    matrix: torch.Tensor  # shape: (... x 4 x 4) K
    pose: torch.Tensor  # shape: (... x 4 x 4) Rt

    @property
    def height(self) -> torch.Tensor:
        return self.get("image_size")[..., 0]

    @property
    def width(self) -> torch.Tensor:
        return self.get("image_size")[..., 1]

    def __post_init__(self):
        if self.matrix.shape[-2:] != (4, 4):
            raise ValueError("Camera matrix must be of shape (..., 4, 4)")
        if self.pose.shape[-2:] != (4, 4):
            raise ValueError("Camera pose must be of shape (..., 4, 4)")
        if self.image_size.shape[-1] != 2:
            raise ValueError("Camera size must be of shape (..., 2)")


class CaptureData(Tensorclass):
    """A capture describes a single frame of data, including images, label maps, depth maps, and camera parameters."""

    times: torch.Tensor
    images: Image
    segmentations: PanopticMap | None
    depths: DepthMap | None

    def __post_init__(self):
        assert (
            self.images.ndim >= 3 and self.images.shape[-3] == 3
        ), f"Images must be of shape (..., 3, H, W), got {self.images.shape}"

    def fix_subtypes_(self) -> T.Self:
        """Subtypes are removed when converting to a tensor, so this method restores them."""

        self.images = self.images.as_subclass(Image)
        self.segmentations = self.segmentations.as_subclass(PanopticMap) if self.segmentations is not None else None
        self.depths = self.depths.as_subclass(DepthMap) if self.depths is not None else None
        return self

    def items(self):
        for k, v in self._tensordict.items():
            if v is None:
                continue

            # HACK: This fixes the issue in tensordict where subclasses are removed for unknown reasons.
            if k == "images":
                v = v.as_subclass(Image)
            elif k == "segmentations":
                v = v.as_subclass(PanopticMap)
            elif k == "depths":
                v = v.as_subclass(DepthMap)

            yield k, v

    @property
    def height(self) -> int:
        """Returns the height of the image."""

        return self.images.shape[-2]

    @property
    def width(self) -> int:
        """Returns the width of the image."""

        return self.images.shape[-1]

    def fillna(self, inplace=True) -> T.Self:
        """
        Materializes the default values for any attribute that is ``None``, i.e. when no data is available for that
        entry. The default shape is inferred from the shape of the images attribute, which may never be ``None`` by
        definition.
        """
        has_panoptic = self.segmentations is not None
        has_depth = self.depths is not None
        if has_panoptic and has_depth:
            return self

        if not inplace:
            self = self.clone()
        shape = torch.Size([*self.batch_size, self.height, self.width])
        device = self.images.device

        if not has_panoptic:
            self.segmentations = PanopticMap.default(shape, device=device)
        if not has_depth:
            self.depths = DepthMap.default(shape, device=device)

        return self


class MotionData(Tensorclass):
    """Data describing motion between time steps."""

    optical_flow: OpticalFlow | None
    transform: torch.Tensor

    def __post_init__(self):
        if self.optical_flow is not None and (
            self.optical_flow.ndim != len(self.batch_size) + 3 or self.optical_flow.shape[-3] != 2
        ):
            raise ValueError("Optical flows must be of shape (..., 2, H, W)")
        if self.transform is not None and (
            self.transform.ndim != len(self.batch_size) + 1 or self.transform.shape[-1] != 4
        ):
            raise ValueError("transform must be of shape (..., 4)")


class InputData(Tensorclass):
    """Describes the input data to any model in the UniPercept framework."""

    ids: torch.Tensor  # N
    captures: CaptureData  # N
    motions: MotionData | None  # N
    cameras: CameraModel  # N
    content_boxes: torch.Tensor = field(
        metadata={
            "help": (
                "Bounding boxes that describe the content of the image. "
                "Useful in cases where some of the images in a batch "
                "are padded, e.g. when using a sampler that samples from a dataset with images of different sizes."
            )
        }
    )

    def __post_init__(self):
        assert (
            len(self.batch_size) <= 1
        ), f"Input data should be unbatched or have one batch dimension got {self.batch_size}"

    def fix_subtypes_(self) -> T.Self:
        self.captures.fix_subtypes_()

        return self

    @classmethod
    def collate(cls, batch: T.Sequence[T.Self]) -> T.Self:
        """Collates a batch of input data into a single input data object."""

        if len(batch) == 0:
            raise ValueError("Batch must be non-empty!")

        assert all(
            len(b.batch_size) == 0 for b in batch
        ), f"Batch size must be 0 for all inputs! Got: {[b.batch_size for b in batch]}"

        return torch.stack(batch)
        # batch_td = [b._tensordict for b in batch]  # type: ignore
        # batch_non_td = [b._non_tensordict for b in batch]  # type:ignore

        # # Convert TensorDict-like attributes
        # td_stack = torch.stack(batch_td)

        # # Convert non-TensorDict attributes (must be None or a sequence)
        # assert all(isinstance(v, T.Sequence) or v is None for n in batch_non_td for v in n.values()), batch_non_td
        # stacked_non_td = {}
        # for n in batch_non_td:
        #     for k, v in n.items():
        #         if v is None:
        #             if k not in stacked_non_td:
        #                 stacked_non_td[k] = None
        #             else:
        #                 assert stacked_non_td[k] is None, f"Expected None, got {stacked_non_td[k]}"
        #             continue
        #         else:
        #             l = stacked_non_td.setdefault(k, [])
        #             assert l is not None, f"Expected list, got {l} at key {k}"
        #             l.extend(v)

        # stacked = cls.from_tensordict(td_stack, non_tensordict=stacked_non_td)
        # assert stacked.batch_size[0] == len(batch), f"Batch size must be {len(batch)}, got {stacked.batch_size[0]}"
        # stacked.refine_names("B")
        # return stacked


class ModelOutput(Tensorclass):
    """
    Describes the output of a model. Standardized in order to work with the training and evaluation
    frameworks.
    """

    losses: TensorDict = field(
        default_factory=lambda: TensorDict({}, batch_size=[]),
        metadata={"help": "Losses to be optimized, should be singular values."},
    )
    metrics: TensorDict = field(
        default_factory=lambda: TensorDict({}, batch_size=[]),
        metadata={"help": "Evaluation metrics, should be singular values."},
    )
    predictions: TensorDict = field(
        default_factory=lambda: TensorDict({}, batch_size=[]), metadata={"help": "Model predictions, can be any shape."}
    )
    truths: TensorDict = field(
        default_factory=lambda: TensorDict({}, batch_size=[]), metadata={"help": "Ground truths for each prediction"}
    )

    def __post_init__(self):
        if isinstance(self.losses, dict):
            self.losses = TensorDict.from_dict(self.losses)
        if isinstance(self.metrics, dict):
            self.metrics = TensorDict.from_dict(self.metrics)
        if isinstance(self.predictions, dict):
            self.predictions = TensorDict.from_dict(self.predictions)
        if isinstance(self.truths, dict):
            self.truths = TensorDict.from_dict(self.truths)


class ModelBase(nn.Module):
    """
    Defines the interface for a perception model.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    @override
    def forward(self, inputs: InputData) -> ModelOutput:
        ...
