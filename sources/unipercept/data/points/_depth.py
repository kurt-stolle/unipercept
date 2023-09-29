from __future__ import annotations

from typing import Self, cast

import torch
from torch.types import Device
from torchvision.datapoints import Mask

from .registry import pixel_maps

__all__ = ["DepthMap"]


@pixel_maps.register
class DepthMap(Mask):
    def __new__(cls, data: torch.Tensor, *args, **kwds) -> Self:
        self = super().__new__(cls, data, *args, **kwds)

        # Torchvision annotation is always `Mask`. Correct it to `DepthMap` by
        # casting to `Self`.`
        return cast(Self, self)

    @classmethod
    def default_like(cls, other: torch.Tensor) -> Self:
        """Returns a default instance of this class with the same shape as the given tensor."""
        return cls(torch.full_like(other, fill_value=0, dtype=torch.float32))

    @classmethod
    def default(cls, shape: torch.Size, device: Device = "cpu") -> Self:
        """Returns a default instance of this class with the given shape."""
        return cls(torch.zeros(shape, device=device, dtype=torch.float32))  # type: ignore
