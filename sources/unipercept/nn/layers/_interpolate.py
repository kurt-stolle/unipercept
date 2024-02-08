"""Implements interpolation layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import typing_extensions as TX


class Interpolate2d(nn.Module):
    r"""
    Wraps :func:`torch.nn.functional.interpolate` as a module.
    """
    name: torch.jit.Final[str]
    size: torch.jit.Final[int | tuple[int, int] | None]
    scale_factor: torch.jit.Final[float | tuple[float, float] | None]
    mode: torch.jit.Final[str]
    align_corners: torch.jit.Final[bool | None]

    def __init__(
        self,
        *,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
        mode: str = "nearest",
        align_corners: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            assert len(scale_factor) == 2, "scale_factor must be a 2-tuple"
            self.scale_factor = (float(scale_factor[0]), float(scale_factor[1]))
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == "nearest" else align_corners

    @TX.override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=False,
        )
