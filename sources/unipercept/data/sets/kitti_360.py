"""
KITTI-360 dataset.

Documentation: https://www.cvlibs.net/datasets/kitti-360/documentation.php
"""

from __future__ import annotations

import dataclasses as D
import re
import typing as T
from pathlib import Path

from tqdm import tqdm

from unipercept import file_io
from unipercept.data.pseudolabeler import PseudoGenerator
from unipercept.data.sets._base import Metadata, PerceptionDataset, create_metadata
from unipercept.data.sets.cityscapes import CLASSES
from unipercept.data.types import (
    CaptureRecord,
    CaptureSources,
    Manifest,
    ManifestSequence,
    PinholeModelParameters,
)


@D.dataclass(frozen=True, slots=True)
class FileID:
    """
    Unique representation of each file in the dataset. Files are organized as the following examples:

        - /datasets/kitti-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rgb/0000000000.png
        - /datasets/kitti-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rgb/0000000001.png
        - /datasets/kitti-360/data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/semantic/0000000000.png
        - /datasets/kitti-360/data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/instance/0000000000.png
    """

    drive: str
    camera: str
    kind: str
    frame: str
    ext: str

    pattern: T.ClassVar[re.Pattern[str]] = re.compile(
        r"(?P<drive>\d\d\d\d_\d\d_\d\d_drive_\d\d\d\d_sync)[\\/]"
        r"(?P<camera>image_\d\d)[\\/]"
        r"(?P<kind>[\w_]+)[\\/]"
        r"(?P<frame>\d+)"
        r"(?P<ext>\..+)$"  # noqa: 501
    )

    @classmethod
    def from_path(cls, path: str) -> T.Self:
        match = cls.pattern.search(path)
        assert match is not None
        return cls(
            drive=match.group("drive"),
            camera=match.group("camera"),
            kind=match.group("kind"),
            frame=match.group("frame"),
            ext=match.group("ext"),
        )

    @classmethod
    def attach_id(cls, path: str) -> tuple[T.Self, str]:
        """Transforms a path into an ID and a dictionary of paths indexed by key."""
        return cls.from_path(path), path


def get_info() -> Metadata:
    return create_metadata(CLASSES, depth_max=80, fps=15)


class KITTI360Dataset(PerceptionDataset, info=get_info, id="kitti-360"):
    split: T.Literal["train", "val"]
    root: str = "//datasets/kitti-360"

    @property
    def root_path(self) -> Path:
        return file_io.Path(self.root)

    def _discover_files(self) -> T.Iterable[FileID]:
        """
        We read the files in each split from a file at:

            - ``{root}/data_2d_semantics/train/2013_05_28_drive_{train/val}_frames.txt``

        The file contains lines formatted as: ``{path_image} {path_semantic}``. We only read the image path.
        """
        with open(
            self.root_path
            / "data_2d_semantics"
            / "train"
            / f"2013_05_28_drive_{self.split}_frames.txt"
        ) as f:
            for line in f:
                file_path = line.split(" ")[0]
                yield FileID.from_path(file_path)

    def _discover_sources(self) -> T.Mapping[FileID, CaptureSources]:
        sources_map: dict[FileID, CaptureSources] = {}
        # Create mapping of ID -> dt.CaptureSources
        files_list = sorted(
            self._discover_files(), key=lambda id: (id.drive, id.camera, id.frame)
        )
        with PseudoGenerator() as pseudo_gen:
            for id in tqdm(files_list, desc="Discovering and generating pseudolabels"):
                if id.kind != "data_rect":
                    continue
                image_path = (
                    self.root_path
                    / "data_2d_raw"
                    / id.drive
                    / id.camera
                    / id.kind
                    / f"{id.frame}{id.ext}"
                )
                assert image_path.is_file(), f"File {image_path} does not exist"

                panseg_path = (
                    self.root_path
                    / "data_2d_semantics"
                    / "train"
                    / id.drive
                    / id.camera
                    / "instance"
                    / f"{id.frame}.png"
                )

                if not panseg_path.is_file():
                    raise RuntimeError(f"File {panseg_path} does not exist")

                depth_path = (
                    self.root_path
                    / "data_2d_depth"
                    / "train"
                    / id.drive
                    / id.camera
                    / "mono_depth"
                    / f"{id.frame}.tiff"
                )
                if not depth_path.is_file():
                    pseudo_gen.add_depth_generator_task(image_path, depth_path)

                partial_sources: CaptureSources = {
                    "image": {
                        "path": image_path.as_posix(),
                    },
                    "panoptic": {
                        "path": panseg_path.as_posix(),
                        "meta": {
                            "format": "vistas",
                        },
                    },
                    "depth": {
                        "path": depth_path.as_posix(),
                        "meta": {
                            "format": "tiff",
                        },
                    },
                }
                sources_map[id] = partial_sources

        if len(sources_map) == 0:
            raise RuntimeError("No files were discovered!")

        return sources_map

    def _get_camera_intrinsics(self, id: FileID) -> PinholeModelParameters:
        """Camera intrinsics are stored in ``{root}/data_2d_raw/{id.drive}/calib_cam_to_cam.txt``."""

        # TODO: return something that is not a stub
        camera_intrinsics: PinholeModelParameters = {
            "focal_length": [0.0, 0.0],
            "image_size": [512, 1024],
            "principal_point": [0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
        }

        return camera_intrinsics

    def _build_manifest(self) -> Manifest:
        sources_map = self._discover_sources()

        # Create the sequences map
        sequences: dict[str, ManifestSequence] = {}
        for id, sources in tqdm(sources_map.items(), desc="Building manifest"):
            if id.kind != "data_rect":
                continue

            cap: CaptureRecord = {
                "primary_key": f"{id.drive}/{id.camera}/{id.frame}",
                "time": float(id.frame),
                "observer": f"{id.camera}",
                "meta": {},
                "sources": sources,
            }

            seq_key = f"{id.drive}"
            seq = sequences.setdefault(
                seq_key,
                ManifestSequence(
                    camera=self._get_camera_intrinsics(id),
                    fps=15.0,
                    motions=[],
                    captures=[],
                ),
            )
            seq["captures"].append(cap)

        # Sort each sequence's captures and motions by time
        for seq in sequences.values():
            seq["captures"].sort(key=lambda c: c["time"])
            seq["motions"].sort(key=lambda m: m["frames"][0])

        return {"sequences": sequences, "timestamp": "2023-08-15", "version": "1.0.0"}
