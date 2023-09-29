"""
KITTI STEP dataset
"""
"""
Depth-Aware Cityscapes and VPS extension.
"""

import functools
import re
from pathlib import Path
from typing import Iterable, NamedTuple

from ..structures import CameraSpec, FileResource, Record
from ._meta import Metadata, generate_metadata
from ._pipes import get_image_size
from .Wrapper import DatasetWrapper

__all__ = ["KITTISTEP"]


class _KITTIMixin:
    pass


class KITTISTEP(_KITTIMixin, DatasetWrapper, split_names={"train", "val", "test"}):
    """
    Implements common functionality for Cityscapes datasets.
    """

    @classmethod
    def get_root(cls) -> Path:
        return super().get_root() / "kitti-step"

    @property
    def labels_path(self) -> Path:
        return self.get_root() / "panoptic_maps" / self.split

    @property
    def depths_path(self) -> Path:
        return self.get_root() / "proj_depth" / self.split

    @property
    def images_path(self) -> Path:
        return self.get_root() / self.split

    def discover(self):
        # Images in `image_dir` are organized as `{seq_id}/{frame_id}.png`
        for seq_path in self.images_path.iterdir():
            if not seq_path.is_dir():
                raise ValueError(f"Not a directory: {seq_path}")

            seq_id = seq_path.name

            for frame_path in seq_path.iterdir():
                frame_id = frame_path.name.replace(".png", "")
                width, height = get_image_size(frame_path)

                record = Record(
                    image_id=frame_id,
                    frame=int(frame_id),
                    sequence_id=seq_id,
                    image=FileResource(path=frame_path.as_posix(), type="image"),
                    camera=CameraSpec(
                        intrinsic={
                            "fx": 707.09,
                            "fy": 707.09,
                            "u0": float(width / 2),
                            "v0": float(height / 2),
                        },
                        extrinsic={},
                    ),
                    height=height,
                    width=width,
                )

                panseg_path = self.labels_path / seq_id / f"{frame_id}.png"
                if panseg_path.is_file():
                    record["panseg"] = FileResource(path=panseg_path.as_posix(), type="kitti")

                depth_path = self.depths_path / seq_id / f"{frame_id}.png"
                if depth_path.is_file():
                    record["depth"] = FileResource(path=depth_path.as_posix(), type="depth")

                yield record

    @functools.cached_property
    def metadata(self) -> Metadata:
        categories = [
            {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
            {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
            {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
            {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
            {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
            {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
            {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
            {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
            {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
            {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
            {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
            {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
            {"color": (255, 0, 0), "isthing": 0, "id": 25, "trainId": 12, "name": "rider"},
            {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
            {"color": (0, 0, 70), "isthing": 0, "id": 27, "trainId": 14, "name": "truck"},
            {"color": (0, 60, 100), "isthing": 0, "id": 28, "trainId": 15, "name": "bus"},
            {"color": (0, 80, 100), "isthing": 0, "id": 31, "trainId": 16, "name": "train"},
            {"color": (0, 0, 230), "isthing": 0, "id": 32, "trainId": 17, "name": "motorcycle"},
            {"color": (119, 11, 32), "isthing": 0, "id": 33, "trainId": 18, "name": "bicycle"},
        ]

        return generate_metadata(
            categories=categories,  # type: ignore
            depth_max=80.0,
            fps=15,
        )
