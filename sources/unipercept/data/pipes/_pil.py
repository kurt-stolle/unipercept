"""
Implements a PIL image loader dataset.
"""

from __future__ import annotations

import os
import typing as T
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image as PILImage

_SupportsPILImageOpen: T.TypeAlias = str | Path | bytes

__all__ = ["PILImageLoaderDataset"]


class PILImageLoaderDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list[_SupportsPILImageOpen]):
        self.image_paths = image_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> PILImage.Image:
        return PILImage.open(self.image_paths[index])
