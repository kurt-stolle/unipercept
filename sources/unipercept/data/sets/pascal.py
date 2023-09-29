"""
Pascal VOC dataset, with panoptic segmentation.

See: https://pytorch.org/vision/0.10/_modules/torchvision/datasets/voc.html#VOCSegmentation
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping, Sequence

import torch
import unipercept.data.io as data_io
import unipercept.data.points as data_points
import unipercept.data.types as data_types
from typing_extensions import override
from unicore import catalog, file_io

from ._base import PerceptionDataset
from ._meta import generate_metadata

__all__ = ["VOC"]


# ----------- #
# Static info #
# ----------- #
def get_info():
    from ..types import RGB, SClass, SType

    class_names = [
        "void",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    ]

    from unipercept.render._colormap import colormap

    cmap = colormap(rgb=True, maximum=255)

    sem_list = []
    for i, name in enumerate(class_names):
        sem_list.append(
            SClass(
                color=RGB(
                    *cmap[i], kind=SType.THING if i > 0 else SType.VOID, dataset_id=i, unified_id=i - 1, name=name
                )
            )
        )

    return generate_metadata(
        sem_list,
        depth_max=100.0,
        fps=15.0,
    )


def _create_panoptic_source(*, seg_path: Path, ins_path: Path, pan_path: Path) -> None:
    import numpy as np
    from PIL import Image

    logging.debug(f"Creating VOC panoptic map: {pan_path.name}")

    assert seg_path.is_file(), f"Expected {seg_path} to exist!"
    assert ins_path.is_file(), f"Expected {ins_path} to exist!"
    assert not pan_path.is_file(), f"Expected {pan_path} to not exist!"

    seg = np.asarray(Image.open(seg_path))
    ins = np.asarray(Image.open(ins_path))
    assert seg.size == ins.size, f"Expected same size, got {seg.size} and {ins.size}!"
    assert seg.ndim == ins.dim == 2, f"Expected 2D images, got {seg.ndim}D and {ins.ndim}D!"

    pan = data_points.PanopticMap.from_parts(semantic=seg, instance=ins)
    torch.save(pan, file_io.get_local_path(pan_path))


# -------------------- #
# Dataset and variants #
# -------------------- #
@catalog.register_dataset("voc")
class VOC(PerceptionDataset, info=get_info):
    root: str = "//datasets/voc"
    split: Literal["train", "val", "trainval", "test"] = "train"
    download: bool = False
    year: str = "2012"

    def _build_panoptic(self) -> list[tuple[str, str]]:
        from torchvision.datasets import VOCSegmentation

        root = file_io.get_local_path(self.root)

        dataset = VOCSegmentation(root, year=self.year, image_set=self.split, download=self.download)
        src_img: list[str] = dataset.images
        src_seg: list[str] = dataset.targets
        src_pan = [
            t.replace("SegmentationClass", "SegmentationPanoptic").replace(dataset._TARGET_FILE_EXT, ".pth")
            for t in src_seg
        ]
        del dataset

        for i in range(len(src_img)):
            pan_path = Path(src_pan[i])
            if file_io.exists(pan_path):
                continue
            seg_path = Path(src_seg[i])
            ins_path = Path(src_seg[i].replace("SegmentationClass", "SegmentationObject"))

            _create_panoptic_source(seg_path=seg_path, ins_path=ins_path, pan_path=pan_path)

        return list(zip(src_img, src_pan))

    @override
    def _build_manifest(self) -> data_types.Manifest:
        paths = self._build_panoptic()

        # Convert to mapping of string -> dt.CaptureRecord
        sequences: Mapping[str, data_types.ManifestSequence] = {}
        for i, (img_path, pan_path) in enumerate(paths):
            captures: list[data_types.CaptureRecord] = [
                {
                    "primary_key": str(i),
                    "sources": {
                        "image": {
                            "path": str(img_path),
                        },
                        "panoptic": {
                            "path": str(pan_path),
                            "type": "torch",
                        },
                    },
                }
            ]

            # Create sequence item
            seq_item: data_types.ManifestSequence = {
                "camera": {
                    "focal_length": (1, 1),
                    "principal_point": (1, 1),
                    "rotation": (0, 0, 0),
                    "translation": (0, 0, 0),
                    "image_size": (512, 512),
                },
                "fps": 15,
                "captures": captures,
            }
            sequences[i] = seq_item

        return data_types.Manifest(
            timestamp=datetime.utcnow().isoformat(), version=f"{self.year}-panoptic", sequences=sequences
        )

    @classmethod
    @override
    def _load_capture_data(
        cls, sources: Sequence[data_types.CaptureSources], info: data_types.Metadata
    ) -> data_points.CaptureData:
        num_caps = len(sources)
        times = torch.linspace(0, num_caps / info["fps"], num_caps)

        return data_points.CaptureData(
            times=times,
            images=data_io.utils.multi_read(
                data_io.read_image,
                key="image",
                no_entries="error",
            )(sources),
            segmentations=data_io.utils.multi_read(
                data_io.read_panoptic_map,
                "panoptic",
                no_entries="none",
            )(sources, info),
            depths=data_io.utils.multi_read(
                data_io.read_depth_map,
                "depth",
                no_entries="none",
            )(sources),
            batch_size=[num_caps],
        )

    @classmethod
    @override
    def _load_motion_data(
        cls, sources: Sequence[data_types.MotionSources], info: data_types.Metadata
    ) -> data_points.MotionData:
        raise NotImplementedError("VOC does not implement motion sources!")
