"""
KITTI STEP dataset
"""

from __future__ import annotations

import typing as T

from unicore import catalog, file_io
from unicore.utils.image import size as get_image_size

from ._base import PerceptionDataset, info_factory, RGB, SClass, SType

__all__ = ["KITTISTEPDataset"]


CLASSES: T.Final[T.Sequence[SClass]] = [
    SClass(color=RGB(128, 64, 128), kind=SType.VOID, dataset_id=255, unified_id=-1, name="void"),
    SClass(color=RGB(128, 64, 128), kind=SType.STUFF, dataset_id=0, unified_id=0, name="road"),
    SClass(color=RGB(244, 35, 232), kind=SType.STUFF, dataset_id=1, unified_id=1, name="sidewalk"),
    SClass(color=RGB(70, 70, 70), kind=SType.STUFF, dataset_id=2, unified_id=2, name="building"),
    SClass(color=RGB(102, 102, 156), kind=SType.STUFF, dataset_id=3, unified_id=3, name="wall"),
    SClass(color=RGB(190, 153, 153), kind=SType.STUFF, dataset_id=4, unified_id=4, name="fence"),
    SClass(color=RGB(153, 153, 153), kind=SType.STUFF, dataset_id=5, unified_id=5, name="pole"),
    SClass(color=RGB(250, 170, 30), kind=SType.STUFF, dataset_id=6, unified_id=6, name="traffic light"),
    SClass(color=RGB(220, 220, 0), kind=SType.STUFF, dataset_id=7, unified_id=7, name="traffic sign"),
    SClass(color=RGB(107, 142, 35), kind=SType.STUFF, dataset_id=8, unified_id=8, name="vegetation"),
    SClass(color=RGB(152, 251, 152), kind=SType.STUFF, dataset_id=9, unified_id=9, name="terrain"),
    SClass(color=RGB(70, 130, 180), kind=SType.STUFF, dataset_id=10, unified_id=10, name="sky", depth_fixed=1.0),
    SClass(color=RGB(220, 20, 60), kind=SType.THING, dataset_id=1, unified_id=11, name="person"),
    SClass(color=RGB(255, 0, 0), kind=SType.STUFF, dataset_id=12, unified_id=12, name="rider"),
    SClass(color=RGB(0, 0, 142), kind=SType.THING, dataset_id=13, unified_id=13, name="car"),
    SClass(color=RGB(0, 0, 70), kind=SType.STUFF, dataset_id=14, unified_id=14, name="truck"),
    SClass(color=RGB(0, 60, 100), kind=SType.STUFF, dataset_id=15, unified_id=15, name="bus"),
    SClass(color=RGB(0, 80, 100), kind=SType.STUFF, dataset_id=16, unified_id=16, name="train"),
    SClass(color=RGB(0, 0, 230), kind=SType.STUFF, dataset_id=17, unified_id=17, name="motorcycle"),
    SClass(color=RGB(119, 11, 32), kind=SType.STUFF, dataset_id=18, unified_id=18, name="bicycle"),
]


def get_info():
    return info_factory(
        CLASSES,
        depth_max=80.0,
        fps=17.0,
    )


class KITTISTEPDataset(PerceptionDataset, info=get_info, id="kitti-step"):
    """Implements KITTI-STEP."""

    root = "//datasets/kitti-step"
    split: T.Literal["train", "val", "test"]

    @property
    def root_path(self) -> file_io.Path:
        return file_io.Path(self.root)

    @property
    def labels_path(self) -> file_io.Path:
        return self.root_path / "panoptic_maps" / self.split

    @property
    def depths_path(self) -> file_io.Path:
        return self.root_path / "proj_depth" / self.split

    @property
    def images_path(self) -> file_io.Path:
        return self.root_path / self.split

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
