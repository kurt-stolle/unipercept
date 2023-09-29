"""Implements a baseclass for the UniPercept dataset common format."""

from __future__ import annotations

import abc
import dataclasses
import typing as T

import torch
from typing_extensions import override
from unicore.utils.dataset import Dataset as _BaseDataset
from uniutils.camera import build_calibration_matrix

from ..collect import ExtractIndividualFrames, QueueGeneratorType
from ..points import CameraModel, CaptureData, InputData, MotionData
from ..types import CaptureSources, Manifest, Metadata, MotionSources, QueueItem

__all__ = ["PerceptionDataset"]


class PerceptionDataset(
    _BaseDataset[Manifest, QueueItem, InputData, Metadata],
):
    """Baseclass for datasets that are composed of captures and motions."""

    queue_fn: T.Callable[[Manifest], QueueGeneratorType] = dataclasses.field(default_factory=ExtractIndividualFrames)

    @classmethod
    @abc.abstractmethod
    def _load_capture_data(cls, sources: T.Sequence[CaptureSources], info: Metadata) -> CaptureData:
        """Loads the capture data from the given captures."""

        raise NotImplementedError(f"{cls.__name__} does not implement `_load_capture_data`!")

    @classmethod
    @abc.abstractmethod
    def _load_motion_data(cls, sources: T.Sequence[MotionSources], info: Metadata) -> MotionData:
        """Loads the motion data from the given motions."""

        raise NotImplementedError(f"{cls.__name__} does not implement `_load_motion_data`!")

    @classmethod
    @override
    def _load_data(cls, key: str, item: QueueItem, info: Metadata) -> InputData:
        # types.utils.check_typeddict(item, QueueItem)

        # Captures
        item_caps = item["captures"]
        item_caps_num = len(item_caps)
        assert item_caps_num > 0

        for cap in item_caps:
            # types_utils.check_typeddict(cap, CaptureSources)
            pass

        data_caps = cls._load_capture_data(item_caps, info)

        # Motions
        if "motions" in item:
            item_mots = item["motions"]
            item_mots_num = len(item_mots)
            assert item_mots_num > 0

            for mot in item_mots:
                # types_utils.check_typeddict(mot, MotionSources)
                pass

            data_mots = cls._load_motion_data(item_mots, info)
        else:
            data_mots = None

        # Camera
        item_camera = item["camera"]
        data_camera = CameraModel(
            matrix=build_calibration_matrix(
                focal_lengths=[item_camera["focal_length"]],
                principal_points=[item_camera["principal_point"]],
                orthographic=False,
            ),
            image_size=torch.as_tensor(item_camera["image_size"]),
            pose=torch.eye(4),
            batch_size=[],
        )

        input_data = InputData(
            captures=data_caps,
            motions=data_mots,
            cameras=data_camera,
            ids=[key],
            batch_size=[],
        )

        return input_data
