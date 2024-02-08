"""
Semantic KITTI and variants.

Expects the dataset to be installed and in the following format:
``
$UNICORE_DATA
    |── semkitti-dvps
    │   ├── video_sequence
    │   │   ├── train
    │   │   │   ├── 000000_000000_leftImg8bit.png
    │   │   │   ├── 000000_000000_gtFine_class.png
    │   │   │   ├── 000000_000000_gtFine_instance.png
    │   │   │   ├── 000000_000000_depth_718.8560180664062.png
    │   │   │   ├── ...
    │   │   ├── val
    │   │   │   ├── ...
``
"""

from __future__ import annotations

import typing as T
import zipfile

from tqdm import tqdm
from typing_extensions import override

from unipercept import file_io
from unipercept.data.pseudolabeler import PseudoGenerator
from unipercept.data.sets._base import (
    RGB,
    Metadata,
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
)
from unipercept.utils.time import get_timestamp

__all__ = ["SemKITTIDataset"]

DEFAULT_URL = (
    "https://huggingface.co/HarborYuan/PolyphonicFormer/resolve/main/semkitti-dvps.zip"
)
CAPTURE_FPS = 17.0


def get_info() -> Metadata:
    sem_list = [
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
            color=RGB(0, 0, 142),
            kind=SType.THING,
            dataset_id=26,
            unified_id=13,
            name="car",
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

    return create_metadata(
        sem_list,
        depth_max=80.0,
        fps=17.0,
    )


class SemKITTIDataset(PerceptionDataset, info=get_info, id="kitti-dvps"):
    """
    Implements the SemKITTI-DVPS dataset introduced by *ViP-DeepLab: [...]* (Qiao et al, 2021).

    Paper: https://arxiv.org/abs/2106.10867
    """

    split: T.Literal["train", "val"]
    root: str = "//datasets/semkitti-dvps"
    pseudo: bool = True
    download: bool = False

    def __post_init__(self):
        if not file_io.isdir(self.root):
            self._download_and_extract()

    def _download_and_extract(self, url: str = DEFAULT_URL) -> None:
        """
        Download and extract the dataset.

        The default download URL is provided by the authors of PolyphonicFormer.
        """

        archive_path = file_io.get_local_path(url)
        with zipfile.ZipFile(archive_path) as zip:
            zip.extractall(self.root)
        file_io.rm(archive_path)

    @override
    def _build_manifest(self) -> Manifest:
        cap_root = file_io.Path(self.root) / "video_sequence" / self.split
        assert cap_root.exists(), f"Captures path {cap_root} does not exist!"

        sequences: dict[str, ManifestSequence] = {}

        pseudo = PseudoGenerator()

        for cap_path in tqdm(
            sorted(cap_root.glob("**/*_leftImg8bit.png"), key=lambda p: p.stem),
            desc="Discovering captured source images",
        ):
            seq_name, frame_name, *_ = cap_path.stem.split("_")
            key = f"{seq_name}_{frame_name}"

            # Ensure that the sequence exists
            seq: ManifestSequence = sequences.setdefault(
                seq_name,
                {
                    "camera": None,
                    "fps": CAPTURE_FPS,
                    "captures": [],
                    "motions": [],
                },
            )

            # Create sources map
            sources: CaptureSources = {
                "image": {
                    "path": str(cap_path),
                }
            }

            # Depth has the focal length encoded in the name, so we must do a search for it
            depth_path = next(cap_path.parent.glob(f"{key}_depth_*.png"), None)
            if depth_path is not None:
                sources["depth"] = {
                    "path": str(depth_path),
                    "meta": {
                        "format": "uint16",
                        "focal_length": float(depth_path.stem.split("_")[-1]),
                    },
                }

            # Panoptic must potentially be generated from the semantic and instance masks
            panoptic_path = cap_path.parent / f"{key}_panoptic.png"
            if not panoptic_path.exists():
                semantic_path = cap_path.parent / f"{key}_gtFine_class.png"
                instance_path = cap_path.parent / f"{key}_gtFine_instance.png"

                pseudo.add_panoptic_merge((semantic_path, instance_path), panoptic_path)

            sources["panoptic"] = {
                "path": str(panoptic_path),
                "meta": {"format": "torch"},
            }

            # Create the capture record
            rec: CaptureRecord = {"primary_key": key, "sources": sources}

            # Add the record to the sequence
            seq["captures"].append(rec)

        return {"timestamp": get_timestamp(), "version": "0.1", "sequences": sequences}
