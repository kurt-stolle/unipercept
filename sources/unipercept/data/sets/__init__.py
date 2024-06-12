"""This module contains the dataset modules."""

from __future__ import annotations

from ._meta import *
from ._base import *
from . import (
    cityscapes,
    huggingface,
    kitti_360,
    kitti_sem,
    kitti_step,
    pascal_voc,
    vistas,
    wilddash,
    coco,
    pattern,
)
