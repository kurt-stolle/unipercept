"""Datapoints for the inputs and outputs of a model."""

from __future__ import annotations

import typing as T

import torch
import torch.utils._pytree as pytree
from tensordict import LazyStackedTensorDict
from torchvision.datapoints import Image
from unicore.utils.tensorclass import Tensorclass, TypedTensorDict

from ._camera import CameraModel
from ._depth import DepthMap
from ._flow import OpticalFlow
from ._panoptic import PanopticMap

__all__ = ["CaptureData", "MotionData", "InputData"]


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

    captures: CaptureData  # N
    motions: MotionData | None  # N
    cameras: CameraModel  # N
    ids: list[str]

    def __post_init__(self):
        assert len(self.batch_size) <= 2, self.batch_size
        # pass

    def fix_subtypes_(self) -> T.Self:
        self.captures.fix_subtypes_()

        return self

    @classmethod
    def collate(cls, batch: T.Sequence[T.Self]) -> T.Self:
        """Collates a batch of input data into a single input data object."""

        if len(batch) == 0:
            raise ValueError("Batch must be non-empty!")

        # non_td = {
        #     k: sum(v) if isinstance(v, T.Sequence) else v for k in ("keys") for v in (getattr(b, k) for b in batch)
        # }

        assert all(
            len(b.batch_size) == 0 for b in batch
        ), f"Batch size must be 0 for all inputs! Got: {[b.batch_size for b in batch]}"

        td = [b._tensordict for b in batch]  # type: ignore
        non_td = [b._non_tensordict for b in batch]  # type:ignore

        assert all(isinstance(v, T.Sequence) for n in non_td for v in n.values())

        stacked_td = LazyStackedTensorDict(*td)
        stacked_non_td = {}
        for n in non_td:
            for k, v in n.items():
                l = stacked_non_td.setdefault(k, [])
                l.extend(v)

        stacked = cls.from_tensordict(stacked_td, non_tensordict=stacked_non_td)
        assert stacked.batch_size[0] == len(batch), f"Batch size must be {len(batch)}, got {stacked.batch_size[0]}"
        stacked.refine_names("B")
