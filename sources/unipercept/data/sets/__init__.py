"""This module contains the dataset modules."""
from __future__ import annotations

import typing as T

import typing_extensions as TX
from typing_extensions import deprecated

from . import (
    cityscapes,
    huggingface,
    kitti_360,
    kitti_sem,
    kitti_step,
    pascal_voc,
    vistas,
    wilddash,
)
from ._base import *
from ._pseudo import *
