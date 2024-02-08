"""
Cityscapes DPS and VPS datasets.
"""

from __future__ import annotations

import dataclasses as D
import operator
import re
import typing as T
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal, Mapping

from typing_extensions import override

from unipercept import file_io
from unipercept.data.pipes import UniCoreFileLister
from unipercept.data.sets._base import (
    RGB,
    PerceptionDataset,
    SClass,
    SType,
    create_metadata,
)
from unipercept.utils.formatter import formatter

if T.TYPE_CHECKING:
    import unipercept as up

__all__ = ["CityscapesDataset", "CityscapesVPSDataset"]


# ---------------- #
# File ID matching #
# ---------------- #


@D.dataclass(frozen=True)
class FileID:
    __slots__ = ("city", "drive", "frame")

    city: str
    drive: str
    frame: str

    pattern: T.ClassVar[re.Pattern[str]] = re.compile(
        r"(?P<city>[A-Za-z]+)_"
        r"(?P<drive>\d\d\d\d\d\d)_"
        r"(?P<frame>\d\d\d\d\d\d)_"
        r"(?P<ext>.+)\..+$"  # noqa: 501
    )

    @classmethod
    def attach_id(cls, path: str) -> tuple[T.Self, str]:
        """
        Transforms a path into an ID and a dictionary of paths indexed by key.
        """

        match = cls.pattern.search(path)
        assert match is not None
        return (
            cls(match.group("city"), match.group("drive"), match.group("frame")),
            path,
        )

    def __lt__(self, other: FileID) -> bool:
        return D.astuple(self) < D.astuple(other)

    def __le__(self, other: FileID) -> bool:
        return D.astuple(self) <= D.astuple(other)

    def __gt__(self, other: FileID) -> bool:
        return D.astuple(self) > D.astuple(other)

    def __ge__(self, other: FileID) -> bool:
        return D.astuple(self) >= D.astuple(other)


def get_primary_key(seq_key: str, idx: int) -> str:
    return f"{seq_key}_{idx:06d}"


def get_sequence_key(seq_idx: int) -> str:
    return f"{seq_idx:06d}"


# ------ #
# Camera #
# ------ #


@dataclass(frozen=True, kw_only=True, slots=True)
class CameraIntrinsic:
    fx: float
    fy: float
    u0: float
    v0: float


@dataclass(frozen=True, kw_only=True, slots=True)
class CameraExtrinsic:
    baseline: float
    pitch: float
    roll: float
    yaw: float
    x: float
    y: float
    z: float


@dataclass(slots=True, frozen=True)
class CameraCalibration:
    """
    See: https://github.com/mcordts/cityscapesScripts/blob/master/docs/csCalibration.pdf
    """

    extrinsic: CameraExtrinsic
    intrinsic: CameraIntrinsic
    size: "up.data.types.HW"

    def to_canonical(self) -> up.data.types.PinholeModelParameters:
        """
        Transforms the calibration to the canonical format.
        """

        return {
            "focal_length": (self.intrinsic.fx, self.intrinsic.fy),
            "principal_point": (self.intrinsic.u0, self.intrinsic.v0),
            "rotation": (self.extrinsic.pitch, self.extrinsic.yaw, self.extrinsic.roll),
            "translation": (self.extrinsic.x, self.extrinsic.y, self.extrinsic.z),
            "image_size": self.size,
        }


CAMERA = CameraCalibration(
    extrinsic=CameraExtrinsic(
        baseline=0.222126, pitch=0.05, roll=0.0, x=1.7, y=-0.1, yaw=0.007, z=1.18
    ),
    intrinsic=CameraIntrinsic(
        fx=2268.36, fy=2225.5405988775956, u0=1048.64, v0=519.277
    ),
    size=(1024, 2048),
)


# ----------- #
# Static info #
# ----------- #

CLASSES: T.Final[T.Sequence[SClass]] = [
    SClass(
        color=RGB(128, 64, 128),
        kind=SType.VOID,
        dataset_id=255,
        unified_id=-1,
        name="void",
    ),
    SClass(
        color=RGB(128, 64, 128),
        kind=SType.STUFF,
        dataset_id=7,
        unified_id=0,
        name="road",
    ),
    SClass(
        color=RGB(244, 35, 232),
        kind=SType.STUFF,
        dataset_id=8,
        unified_id=1,
        name="sidewalk",
    ),
    SClass(
        color=RGB(70, 70, 70),
        kind=SType.STUFF,
        dataset_id=11,
        unified_id=2,
        name="building",
    ),
    SClass(
        color=RGB(102, 102, 156),
        kind=SType.STUFF,
        dataset_id=12,
        unified_id=3,
        name="wall",
    ),
    SClass(
        color=RGB(190, 153, 153),
        kind=SType.STUFF,
        dataset_id=13,
        unified_id=4,
        name="fence",
    ),
    SClass(
        color=RGB(153, 153, 153),
        kind=SType.STUFF,
        dataset_id=17,
        unified_id=5,
        name="pole",
    ),
    SClass(
        color=RGB(250, 170, 30),
        kind=SType.STUFF,
        dataset_id=19,
        unified_id=6,
        name="traffic light",
    ),
    SClass(
        color=RGB(220, 220, 0),
        kind=SType.STUFF,
        dataset_id=20,
        unified_id=7,
        name="traffic sign",
    ),
    SClass(
        color=RGB(107, 142, 35),
        kind=SType.STUFF,
        dataset_id=21,
        unified_id=8,
        name="vegetation",
    ),
    SClass(
        color=RGB(152, 251, 152),
        kind=SType.STUFF,
        dataset_id=22,
        unified_id=9,
        name="terrain",
    ),
    SClass(
        color=RGB(70, 130, 180),
        kind=SType.STUFF,
        dataset_id=23,
        unified_id=10,
        name="sky",
        depth_fixed=0.0,
    ),
    SClass(
        color=RGB(220, 20, 60),
        kind=SType.THING,
        dataset_id=24,
        unified_id=11,
        name="person",
    ),
    SClass(
        color=RGB(255, 0, 0),
        kind=SType.THING,
        dataset_id=25,
        unified_id=12,
        name="rider",
    ),
    SClass(
        color=RGB(0, 0, 142), kind=SType.THING, dataset_id=26, unified_id=13, name="car"
    ),
    SClass(
        color=RGB(0, 0, 70),
        kind=SType.THING,
        dataset_id=27,
        unified_id=14,
        name="truck",
    ),
    SClass(
        color=RGB(0, 60, 100),
        kind=SType.THING,
        dataset_id=28,
        unified_id=15,
        name="bus",
    ),
    SClass(
        color=RGB(0, 80, 100),
        kind=SType.THING,
        dataset_id=31,
        unified_id=16,
        name="train",
    ),
    SClass(
        color=RGB(0, 0, 230),
        kind=SType.THING,
        dataset_id=32,
        unified_id=17,
        name="motorcycle",
    ),
    SClass(
        color=RGB(119, 11, 32),
        kind=SType.THING,
        dataset_id=33,
        unified_id=18,
        name="bicycle",
    ),
]


def get_info():
    return create_metadata(
        CLASSES,
        depth_max=80.0,
        fps=17.0,
    )


# -------------------- #
# Dataset and variants #
# -------------------- #


class CityscapesDataset(PerceptionDataset, info=get_info, id="cityscapes"):
    """
    Cityscapes dataset with all data sourced from the official distribution.

    Link: https://www.cityscapes-dataset.com/
    """

    split: Literal["train", "val", "test"]
    root: str = "//datasets/cityscapes"

    path_image = formatter("{self.root}/leftImg8bit/{self.split}")
    path_panoptic = formatter("{self.root}/gtFine/cityscapes_panoptic_{self.split}")
    path_depth = formatter("{self.root}/disparity/{self.split}")
    path_camera = formatter("{self.root}/camera/{self.split}")

    meta_panoptic: T.ClassVar = {"format": "cityscapes"}
    meta_depth: T.ClassVar = {
        "format": "disparity_int16",
        "camera_baseline": CAMERA.extrinsic.baseline,
        "camera_fx": CAMERA.intrinsic.fx,
    }

    def _get_next_frame(self, frame: int) -> int:
        """
        Returns the next frame number from the given frame number.

        Useful for Cityscapes VPS in train mode without the full flag, as only
        every 5th frame will be present.
        """
        return frame + 1

    def _get_id2sources(self) -> Mapping[FileID, up.data.types.CaptureSources]:
        sources_map: dict[FileID, up.data.types.CaptureSources] = {}

        # Create mapping of ID -> dt.CaptureSources
        for id, file_path in map(
            FileID.attach_id,
            UniCoreFileLister(self.path_image, masks="*.png", recursive=True),
        ):
            partial_sources: up.data.types.CaptureSources = {
                "image": {
                    "path": file_path,
                },
            }
            sources_map[id] = partial_sources

        if len(sources_map) == 0:
            raise RuntimeError(f"Found no images in '{self.path_image}'")

        sources_dict = {}
        if file_io.isdir(self.path_panoptic):
            sources_dict["panoptic"] = UniCoreFileLister(
                self.path_panoptic, masks="*.png", recursive=True
            )
        if file_io.isdir(self.path_depth):
            sources_dict["depth"] = UniCoreFileLister(
                self.path_depth, masks="*.png", recursive=True
            )

        for source_key, files in sources_dict.items():
            for id, file_path in map(FileID.attach_id, files):
                if id not in sources_map:
                    raise ValueError(
                        f"File {file_path} (ID: {id}) does not have a corresponding image file, keys: "
                        + ", ".join([str(k) for k in sources_map.keys()])
                    )

                resource: up.data.types.FileResourceWithMeta = {
                    "path": file_path,
                    "meta": {},
                }
                match source_key:
                    case "panoptic":
                        resource["meta"] = self.meta_panoptic
                    case "depth":
                        resource["meta"] = self.meta_depth
                    case _:
                        pass

                sources_map[id][source_key] = resource

        return sources_map

    def _get_seq2ids(self, ids: Iterable[FileID]) -> Mapping[str, list[FileID]]:
        # Convert IDs to a mapping of SequenceID -> list[ID]
        sequence_origins: dict[str, list[FileID]] = {}
        for id in ids:
            sequence_origins.setdefault(f"{id.city}_{id.drive}", []).append(id)

        sequence_map: dict[str, list[FileID]] = {}
        seq_idx = 0
        for origin, ids in sorted(sequence_origins.items(), key=operator.itemgetter(0)):
            assert (
                len(ids) > 0
            ), f"Encountered no IDs in sequence origin map @ '{origin}'"

            ids.sort(key=operator.attrgetter("frame"))

            # It can be that the same origin (city and drive) has multiple sequences
            # (e.g. because the sequence was split into multiple parts). In this case,
            # we need to make sure that the sequences are continuous, i.e. that there
            # are no missing frames between the sequences.

            # Handle the first frame individually
            id = ids.pop(0)
            sequence_map[get_sequence_key(seq_idx)] = [id]
            last_frame = int(id.frame)

            # Handle remaining frames
            for id in ids:
                # Check if the current ID is part of the current sequence
                curr_frame = int(id.frame)
                if self._get_next_frame(last_frame) != curr_frame:
                    seq_idx += 1
                last_frame = curr_frame

                # Insert the ID into the sequence map
                sequence_key = get_sequence_key(seq_idx)
                # sequence_map[sequence_key].append(id)
                sequence_map.setdefault(sequence_key, []).append(id)
            seq_idx += 1

        return sequence_map

    @override
    def _build_manifest(self) -> up.data.types.Manifest:
        sources_map = self._get_id2sources()
        sequence_map = self._get_seq2ids(sorted(sources_map.keys()))

        # Convert to mapping of string -> dt.CaptureRecord
        sequences: Mapping[str, up.data.types.ManifestSequence] = {}
        for seq_key, ids in sequence_map.items():
            camera = CAMERA.to_canonical()  # TODO: read from json
            captures: list[up.data.types.CaptureRecord] = [
                {
                    "primary_key": get_primary_key(seq_key, i),
                    "sources": sources_map[id],
                }
                for i, id in enumerate(ids)
            ]

            # Create sequence item
            seq_item: up.data.types.ManifestSequence = {
                "camera": camera,
                "fps": 17 / self._get_next_frame(0),
                "captures": captures,
            }
            sequences[seq_key] = seq_item

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0",
            "sequences": sequences,
        }


class CityscapesVPSDataset(CityscapesDataset, info=get_info, id="cityscapes-vps"):
    """
    Cityscapes dataset with the following modifications:

    1. **Panoptic maps**
        - Sourced from the paper *Video Panoptic Segmentation* (Kim et al., 2020), which provides a
            panoptic map for every 5th frame.
        - Splits are newly defined within the original Cityscapes distribution's `sequences` subset.
        - Paper: https://arxiv.org/abs/2003.07593
    2. **Depth maps**
        - Sourced from the paper *ViP-DeepLab [...]* (Qiao et al., 2020), which provides a depth map for
            all frames that also have a panoptic label.
        - Same type of disparity-based estimation as original Cityscapes, but with unknown parameters to prevent
            direct comparison.
        - Paper: https://arxiv.org/abs/2012.05258

    """

    split: Literal["train", "val", "test"]
    root: str = "//datasets/cityscapes-vps"
    all: bool = True

    @property
    @override
    def path_image(self):
        img_dir = "img_all" if self.all else "img"
        return f"{self.root}/{self.split}/{img_dir}"

    path_panoptic = formatter("{self.root}/{self.split}/panoptic_inst")
    path_depth = formatter("{self.root}/{self.split}/depth")

    meta_panoptic: T.ClassVar = {"format": "cityscapes_vps"}
    meta_depth: T.ClassVar = {"format": "depth_int16"}

    @property
    @override
    def path_camera(self):
        return f"{super().root}/camera/{self.split}"

    @override
    def _get_next_frame(self, frame: int) -> int:
        """
        Returns the next frame number from the given frame number.

        Useful for Cityscapes VPS in train mode without the full flag, as only
        every 5th frame will be present.
        """
        if self.all:
            return frame + 1
        else:
            return frame + 5
