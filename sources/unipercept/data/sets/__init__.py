"""This module contains the dataset modules."""

from __future__ import annotations

from . import (
    cityscapes,
    coco,
    huggingface,
    kitti_360,
    kitti_sem,
    kitti_step,
    pascal_voc,
    pattern,
    vistas,
    wilddash,
)
from ._base import *
from ._meta import *
