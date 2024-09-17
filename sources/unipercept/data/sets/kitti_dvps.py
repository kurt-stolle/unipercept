"""
Semantic KITTI-DVPS dataset.

Expects the dataset to be installed and in the following format:

.. code-block:: none
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

The number before the ``.png`` extension for the depth label is the focal length of the camera.
"""

from __future__ import annotations

import os
import typing as T
import zipfile
from typing import override

import typing_extensions as TX
from tqdm import tqdm

from unipercept import file_io
from unipercept.utils.time import get_timestamp

from . import (
    RGB,
    CaptureSources,
    Manifest,
    ManifestSequence,
    Metadata,
    PerceptionDataset,
    SClass,
    SType,
)

__all__ = ["KITTIDVPSDataset"]

DOWNLOAD_URL: T.Final = (
    "https://huggingface.co/HarborYuan/PolyphonicFormer/resolve/main/semkitti-dvps.zip"
)
DEFAULT_FOCAL_LENGTH: T.Final = 718.8560180664062
CAPTURE_FPS: T.Final = 17.0


def get_info() -> Metadata:
    sem_list = [
        SClass(
            color=RGB(0, 0, 0),
            kind=SType.VOID,
            dataset_id=255,
            unified_id=-1,
            name="void",
        ),
        SClass(
            color=RGB(245, 150, 100),
            kind=SType.THING,
            dataset_id=0,
            unified_id=0,
            name="car",
        ),
        SClass(
            color=RGB(245, 230, 100),
            kind=SType.THING,
            dataset_id=1,
            unified_id=1,
            name="bicycle",
        ),
        SClass(
            color=RGB(150, 60, 30),
            kind=SType.THING,
            dataset_id=2,
            unified_id=2,
            name="motorcycle",
        ),
        SClass(
            color=RGB(180, 30, 80),
            kind=SType.THING,
            dataset_id=3,
            unified_id=3,
            name="truck",
        ),
        SClass(
            color=RGB(255, 0, 0),
            kind=SType.THING,
            dataset_id=4,
            unified_id=4,
            name="other-vehicle",
        ),
        SClass(
            color=RGB(30, 30, 255),
            kind=SType.THING,
            dataset_id=5,
            unified_id=5,
            name="person",
        ),
        SClass(
            color=RGB(200, 40, 255),
            kind=SType.THING,
            dataset_id=6,
            unified_id=6,
            name="bicyclist",
        ),
        SClass(
            color=RGB(90, 30, 150),
            kind=SType.THING,
            dataset_id=7,
            unified_id=7,
            name="motorcyclist",
        ),
        SClass(
            color=RGB(255, 0, 255),
            kind=SType.STUFF,
            dataset_id=8,
            unified_id=8,
            name="road",
        ),
        SClass(
            color=RGB(255, 150, 255),
            kind=SType.STUFF,
            dataset_id=9,
            unified_id=9,
            name="parking",
        ),
        SClass(
            color=RGB(75, 0, 75),
            kind=SType.STUFF,
            dataset_id=10,
            unified_id=10,
            name="sidewalk",
        ),
        SClass(
            color=RGB(75, 0, 175),
            kind=SType.STUFF,
            dataset_id=11,
            unified_id=11,
            name="other-ground",
        ),
        SClass(
            color=RGB(0, 200, 255),
            kind=SType.STUFF,
            dataset_id=12,
            unified_id=12,
            name="building",
        ),
        SClass(
            color=RGB(50, 120, 255),
            kind=SType.STUFF,
            dataset_id=13,
            unified_id=13,
            name="fence",
        ),
        SClass(
            color=RGB(0, 175, 0),
            kind=SType.STUFF,
            dataset_id=14,
            unified_id=14,
            name="vegetation",
        ),
        SClass(
            color=RGB(0, 60, 135),
            kind=SType.STUFF,
            dataset_id=15,
            unified_id=15,
            name="trunk",
        ),
        SClass(
            color=RGB(80, 240, 150),
            kind=SType.STUFF,
            dataset_id=16,
            unified_id=16,
            name="terrain",
        ),
        SClass(
            color=RGB(150, 240, 255),
            kind=SType.STUFF,
            dataset_id=17,
            unified_id=17,
            name="pole",
        ),
        SClass(
            color=RGB(0, 0, 255),
            kind=SType.STUFF,
            dataset_id=18,
            unified_id=18,
            name="traffic-sign",
        ),
    ]

    return Metadata.from_parameters(
        sem_list,
        depth_max=80.0,
        fps=17.0,
    )


class KITTIDVPSDataset(
    PerceptionDataset, info=get_info, id="kitti-dvps", version="3.0"
):
    """
    Implements the KITTISemanticDVPS dataset introduced by *ViP-DeepLab: [...]* (Qiao et al, 2021).

    Paper: https://arxiv.org/abs/2106.10867
    """

    split: T.Literal["train", "val"]
    root: str = "//datasets/semkitti-dvps"
    pseudo: bool = True

    @classmethod
    @TX.override
    def options(cls) -> dict[str, list[str]]:
        return {
            "split": ["train", "val"],
        }

    @TX.override
    def download(self, *, force: bool = False) -> None:
        """
        Download and extract the dataset.

        The default download URL is provided by the authors of PolyphonicFormer.
        """

        if file_io.is_dir(self.root) and not force:
            return

        archive_path = file_io.get_local_path(DOWNLOAD_URL)
        with zipfile.ZipFile(archive_path) as zip:
            zip.extractall(self.root)
        file_io.rm(archive_path)

    @override
    def _build_manifest(self) -> Manifest:
        from unipercept.utils.image import size as get_image_size

        def _discover_images(
            directory: file_io.Path, endswith: str = "_leftimg8bit.png"
        ) -> T.Iterator[file_io.Path]:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(
                        endswith.lower()
                    ):
                        yield file_io.Path(entry.path)

        cap_root = file_io.Path(self.root) / "video_sequence" / self.split
        assert cap_root.exists(), f"Captures path {cap_root} does not exist!"

        images = sorted(_discover_images(cap_root), key=lambda p: p.stem)

        if len(images) == 0:
            raise RuntimeError(f"No images found in {cap_root}")

        sequences: dict[str, ManifestSequence] = {}
        for cap_path in tqdm(
            images,
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
                focal_length = float(depth_path.stem.split("_")[-1])
                sources["depth"] = {
                    "path": str(depth_path),
                    "meta": {
                        "format": "depth_int16",
                        "focal_length": focal_length,
                    },
                }
            else:
                focal_length = DEFAULT_FOCAL_LENGTH

            # Set the camera of the sequence
            if seq["camera"] is None:
                image_size = get_image_size(cap_path)
                cam: CameraModelParameters = {
                    "focal_length": (focal_length, focal_length),
                    "principal_point": (
                        image_size.height // 2,
                        image_size.width // 2,
                    ),
                    "rotation": (0.0, 0.0, 0.0),
                    "translation": (0.0, 0.0, 0.0),
                    "image_size": (image_size.height, image_size.width),
                    "convention": "iso8855",
                }
                seq["camera"] = cam
            else:
                cam = seq["camera"]
                if not isinstance(cam, dict):
                    msg = f"Camera metadata is not a dictionary: {cam}"
                    raise TypeError(msg)

                assert (
                    cam["focal_length"][0] == cam["focal_length"][1] == focal_length
                ), (
                    f"Camera focal length mismatch: {cam['focal_length']} "
                    f"!= {focal_length}"
                )

            # Panoptic must potentially be generated from the semantic and instance masks
            semantic_path = cap_path.parent / f"{key}_gtFine_class.png"
            sources["semantic"] = {
                "path": str(semantic_path),
                "meta": {"format": "png_l16"},
            }

            instance_path = cap_path.parent / f"{key}_gtFine_instance.png"
            sources["instance"] = {
                "path": str(instance_path),
                "meta": {"format": "png_l16"},
            }

            # Create the capture record
            rec: CaptureRecord = {"primary_key": key, "sources": sources}

            # Add the record to the sequence
            seq["captures"].append(rec)

        return {
            "timestamp": get_timestamp(),
            "version": self.version,
            "sequences": sequences,
        }
