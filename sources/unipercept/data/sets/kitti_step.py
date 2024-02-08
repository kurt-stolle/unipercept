"""
KITTI STEP dataset.

===================

See Also
--------

`cvlibs <https://www.cvlibs.net/datasets/kitti//eval_step.php>`_
"""

from __future__ import annotations

import dataclasses as D
import functools
import re
import typing as T

import typing_extensions as TX
from tqdm import tqdm

from unipercept import file_io
from unipercept.data.pseudolabeler import PseudoGenerator
from unipercept.data.sets._base import (
    RGB,
    PerceptionDataset,
    SClass,
    SType,
    create_metadata,
)
from unipercept.data.types import (
    CaptureRecord,
    CaptureSources,
    Manifest,
    ManifestSequence,
    PinholeModelParameters,
)
from unipercept.log import get_logger

__all__ = ["KITTISTEPDataset"]


_logger = get_logger(__name__)

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
        dataset_id=0,
        unified_id=0,
        name="road",
    ),
    SClass(
        color=RGB(244, 35, 232),
        kind=SType.STUFF,
        dataset_id=1,
        unified_id=1,
        name="sidewalk",
    ),
    SClass(
        color=RGB(70, 70, 70),
        kind=SType.STUFF,
        dataset_id=2,
        unified_id=2,
        name="building",
    ),
    SClass(
        color=RGB(102, 102, 156),
        kind=SType.STUFF,
        dataset_id=3,
        unified_id=3,
        name="wall",
    ),
    SClass(
        color=RGB(190, 153, 153),
        kind=SType.STUFF,
        dataset_id=4,
        unified_id=4,
        name="fence",
    ),
    SClass(
        color=RGB(153, 153, 153),
        kind=SType.STUFF,
        dataset_id=5,
        unified_id=5,
        name="pole",
    ),
    SClass(
        color=RGB(250, 170, 30),
        kind=SType.STUFF,
        dataset_id=6,
        unified_id=6,
        name="traffic light",
    ),
    SClass(
        color=RGB(220, 220, 0),
        kind=SType.STUFF,
        dataset_id=7,
        unified_id=7,
        name="traffic sign",
    ),
    SClass(
        color=RGB(107, 142, 35),
        kind=SType.STUFF,
        dataset_id=8,
        unified_id=8,
        name="vegetation",
    ),
    SClass(
        color=RGB(152, 251, 152),
        kind=SType.STUFF,
        dataset_id=9,
        unified_id=9,
        name="terrain",
    ),
    SClass(
        color=RGB(70, 130, 180),
        kind=SType.STUFF,
        dataset_id=10,
        unified_id=10,
        name="sky",
    ),
    SClass(
        color=RGB(220, 20, 60),
        kind=SType.THING,
        dataset_id=1,
        unified_id=11,
        name="person",
    ),
    SClass(
        color=RGB(255, 0, 0),
        kind=SType.STUFF,
        dataset_id=12,
        unified_id=12,
        name="rider",
    ),
    SClass(
        color=RGB(0, 0, 142), kind=SType.THING, dataset_id=13, unified_id=13, name="car"
    ),
    SClass(
        color=RGB(0, 0, 70),
        kind=SType.STUFF,
        dataset_id=14,
        unified_id=14,
        name="truck",
    ),
    SClass(
        color=RGB(0, 60, 100),
        kind=SType.STUFF,
        dataset_id=15,
        unified_id=15,
        name="bus",
    ),
    SClass(
        color=RGB(0, 80, 100),
        kind=SType.STUFF,
        dataset_id=16,
        unified_id=16,
        name="train",
    ),
    SClass(
        color=RGB(0, 0, 230),
        kind=SType.STUFF,
        dataset_id=17,
        unified_id=17,
        name="motorcycle",
    ),
    SClass(
        color=RGB(119, 11, 32),
        kind=SType.STUFF,
        dataset_id=18,
        unified_id=18,
        name="bicycle",
    ),
]


@D.dataclass(frozen=True, slots=True)
class FileID:
    """
    Unique representation of each captured sample in the dataset.
    Files are organized as the following examples:

        - **/{seq}/{frame}.{ext}
    """

    seq: str
    frame: str
    ext: str

    pattern: T.ClassVar[re.Pattern[str]] = re.compile(
        r"(?P<camera>image_\d\d)[\\/]"
        r"(?P<seq>[\d_]+)[\\/]"
        r"(?P<frame>\d+)"
        r"(?P<ext>\..+)$"  # noqa: 501
    )

    @classmethod
    def from_path(cls, path: str) -> T.Self:
        match = cls.pattern.search(path)
        if match is not None:
            camera = match.group("camera")
            if camera != "image_02":
                msg = f"Camera {camera!r} is not supported"
                raise ValueError(msg)

            return cls(
                seq=match.group("seq"),
                frame=match.group("frame"),
                ext=match.group("ext"),
            )
        else:
            msg = f"Path {path!r} does not match the expected pattern {cls.pattern.pattern!r}"
            raise ValueError(msg)

    @classmethod
    def attach_id(cls, path: str) -> tuple[T.Self, str]:
        """
        Transforms a path into an ID and a dictionary of paths indexed by key.
        """
        return cls.from_path(path), path


class KITTISTEPDataset(
    PerceptionDataset,
    info=functools.partial(create_metadata, CLASSES, depth_max=80, fps=15),
    id="kitti-step",
):
    split: T.Literal["train", "val", "test"]
    root: str = "//datasets/kitti-step"

    @property
    def root_path(self) -> file_io.Path:
        return file_io.Path(self.root)

    @property
    def is_installed(self) -> bool:
        return file_io.isdir(self.root)

    def _discover_files(self) -> T.Iterable[FileID]:
        """
        The dataset has a peculiar structure:

        - {root}
            - training
                - {sequence_training}
                    - {frame}.png  <-- input image
            - testing
                - {sequence_testing}
                    - {frame}.png  <-- input image
            - panoptic_maps
                - train
                    - {sequence_training}
                        - {frame}.png  <-- panoptic segmentation
                - val
                    - {sequence_training}
                        - {frame}.png  <-- panoptic segmentation


        We discover the files by reading the input images and in the case of train/val
        use the panoptic map sequence lists to exclude/include sequences.
        """

        # Read the sequence lists from the panoptic maps
        if self.split != "test":
            seq_list = [
                p.name
                for p in (self.root_path / "panoptic_maps" / self.split).glob("*")
                if p.is_dir()
            ]

            if len(seq_list) == 0:
                msg = (
                    f"No sequences found in {self.root_path} for {self.split}. "
                    "Did you download the panoptic maps?"
                )
                raise RuntimeError(msg)

            _logger.debug("Filtering sequences: include %s", seq_list)
        else:
            seq_list = None

        images_dir = (
            self.root_path
            / ("training" if self.split != "test" else "testing")
            / "image_02"
        )
        if not images_dir.is_dir():
            msg = f"Directory {images_dir} does not exist!"
            raise FileNotFoundError(msg)
        images_paths = images_dir.glob("**/*.png")
        images_paths = (FileID.from_path(p.as_posix()) for p in images_paths)
        if seq_list is not None:
            images_paths = filter(lambda id: id.seq in seq_list, images_paths)

        yield from images_paths

    def _discover_sources(self) -> T.Mapping[FileID, CaptureSources]:
        source_map: dict[FileID, CaptureSources] = {}
        files_list = sorted(self._discover_files(), key=lambda id: (id.seq, id.frame))
        if len(files_list) == 0:
            msg = "Input files not found!"
            raise FileNotFoundError(msg)

        with PseudoGenerator() as pseudo:
            for id in tqdm(files_list, desc="Discovering dataset files"):
                image_path = (
                    self.root_path
                    / ("training" if self.split != "test" else "testing")
                    / "image_02"
                    / id.seq
                    / f"{id.frame}{id.ext}"
                )
                assert image_path.is_file(), f"File {image_path} does not exist"

                panseg_path = (
                    self.root_path
                    / "panoptic_maps"
                    / self.split
                    / id.seq
                    / f"{id.frame}.png"
                )

                if not panseg_path.is_file():
                    panseg_path = None

                depth_path = (
                    self.root_path
                    / "mono_depth"
                    / self.split
                    / id.seq
                    / f"{id.frame}.tiff"
                )
                if not depth_path.is_file():
                    pseudo.add_depth_generator_task(image_path, depth_path)

                partial_sources: CaptureSources = {
                    "image": {
                        "path": image_path.as_posix(),
                    },
                    "depth": {
                        "path": depth_path.as_posix(),
                        "meta": {
                            "format": "tiff",
                        },
                    },
                }
                if panseg_path is not None:
                    partial_sources["panoptic"] = {
                        "path": panseg_path.as_posix(),
                        "meta": {
                            "format": "kitti",
                        },
                    }
                source_map[id] = partial_sources

        if len(source_map) == 0:
            msg = "No files were discovered!"
            raise RuntimeError(msg)

        return source_map

    def _get_camera_intrinsics(self, id: FileID) -> PinholeModelParameters:
        """
        Returns the camera intrinsics for the given sequence and camera.
        """

        # TODO: return something that is not a stub
        camera_intrinsics: PinholeModelParameters = {
            "focal_length": [0.0, 0.0],
            "image_size": [512, 1024],
            "principal_point": [0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
        }

        return camera_intrinsics

    @TX.override
    def _build_manifest(self) -> Manifest:
        sources_map = self._discover_sources()

        # Create the sequences map
        sequences: dict[str, ManifestSequence] = {}
        for id, sources in tqdm(sources_map.items(), desc="Building manifest"):
            seq_key = f"{self.split[:3]}{id.seq}"
            cap: CaptureRecord = {
                "primary_key": f"kitti-step/{seq_key}/{id.frame}",
                "time": float(id.frame) / self.info.fps,  # t = frame / fps
                "observer": "image_02",
                "meta": {},
                "sources": sources,
            }
            seq_key = f"{self.split[:3]}{id.seq}"
            seq = sequences.setdefault(
                seq_key,
                ManifestSequence(
                    camera=self._get_camera_intrinsics(id),
                    fps=self.info.fps,
                    motions=[],
                    captures=[],
                ),
            )
            seq["captures"].append(cap)

        # Sort each sequence's captures and motions by time
        for seq in sequences.values():
            seq["captures"].sort(key=lambda c: c["time"])
            seq["motions"].sort(key=lambda m: m["frames"][0])

        if len(sequences) == 0:
            msg = "No sequences were discovered! Check the dataset installation"
            raise RuntimeError(msg)

        return {"sequences": sequences, "timestamp": "2023-01-30", "version": "1.0.0"}
