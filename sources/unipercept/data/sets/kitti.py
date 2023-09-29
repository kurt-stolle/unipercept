"""KITTI."""

from __future__ import annotations

import typing

from unicore import catalog

from ._base import PerceptionDataset
from ._meta import generate_metadata

__all__ = ["KITTI_STEP"]


def get_info():
    from ..types import RGB, SClass, SType

    sem_list = [
        SClass(color=RGB(128, 64, 128), kind=SType.STUFF, dataset_id=7, unified_id=0, name="road"),
        SClass(color=RGB(244, 35, 232), kind=SType.STUFF, dataset_id=8, unified_id=1, name="sidewalk"),
        SClass(color=RGB(70, 70, 70), kind=SType.STUFF, dataset_id=11, unified_id=2, name="building"),
        SClass(color=RGB(102, 102, 156), kind=SType.STUFF, dataset_id=12, unified_id=3, name="wall"),
        SClass(color=RGB(190, 153, 153), kind=SType.STUFF, dataset_id=13, unified_id=4, name="fence"),
        SClass(color=RGB(153, 153, 153), kind=SType.STUFF, dataset_id=17, unified_id=5, name="pole"),
        SClass(color=RGB(250, 170, 30), kind=SType.STUFF, dataset_id=19, unified_id=6, name="traffic light"),
        SClass(color=RGB(220, 220, 0), kind=SType.STUFF, dataset_id=20, unified_id=7, name="traffic sign"),
        SClass(color=RGB(107, 142, 35), kind=SType.STUFF, dataset_id=21, unified_id=8, name="vegetation"),
        SClass(color=RGB(152, 251, 152), kind=SType.STUFF, dataset_id=22, unified_id=9, name="terrain"),
        SClass(color=RGB(70, 130, 180), kind=SType.STUFF, dataset_id=23, unified_id=10, name="sky"),
        SClass(color=RGB(220, 20, 60), kind=SType.THING, dataset_id=24, unified_id=11, name="person"),
        SClass(color=RGB(255, 0, 0), kind=SType.THING, dataset_id=25, unified_id=12, name="rider"),
        SClass(color=RGB(0, 0, 142), kind=SType.THING, dataset_id=26, unified_id=13, name="car"),
        SClass(color=RGB(0, 0, 70), kind=SType.THING, dataset_id=27, unified_id=14, name="truck"),
        SClass(color=RGB(0, 60, 100), kind=SType.THING, dataset_id=28, unified_id=15, name="bus"),
        SClass(color=RGB(0, 80, 100), kind=SType.THING, dataset_id=31, unified_id=16, name="train"),
        SClass(color=RGB(0, 0, 230), kind=SType.THING, dataset_id=32, unified_id=17, name="motorcycle"),
        SClass(color=RGB(119, 11, 32), kind=SType.THING, dataset_id=33, unified_id=18, name="bicycle"),
    ]

    return generate_metadata(
        sem_list,
        depth_max=80.0,
        fps=17.0,
    )


@catalog.register_dataset("kitti/step")
class KITTI_STEP(PerceptionDataset, info=get_info):
    """Implements KITTI-STEP."""

    root = "//datasets/kitti-step"
    split: typing.Literal["train", "val", "test"]
