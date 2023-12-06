"""
Localizer is a module that takes a single-level feature map and returns the positions of objects
"""

import typing as T

import torch
from torch import nn
from typing_extensions import override

from unipercept.nn.layers import conv
from unipercept.nn.layers.weight import init_xavier_fill_

__all__ = ["Localizer", "Locations"]


class Locations(T.NamedTuple):
    stuff_map: torch.Tensor
    thing_map: torch.Tensor


class Localizer(nn.Module):
    """
    Localizer is a module that takes a single-level feature map and returns the positions of objects
    in the scene as a Gaussian score-map of center locations.
    """

    in_channels: T.Final[int]
    stuff_channels: T.Final[int]
    thing_channels: T.Final[int]

    def __init__(
        self,
        encoder: nn.Module,
        stuff_channels: int,
        thing_channels: int,
        thing_bias_value: float = -2.19,
        squeeze_excite: nn.Module | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.se = squeeze_excite if squeeze_excite is not None else nn.Identity()

        out_enc = encoder.out_channels

        self.stuff_out = conv.Conv2d(out_enc, stuff_channels, kernel_size=3, bias=True, padding=1)
        self.thing_out = conv.Conv2d(out_enc, thing_channels, kernel_size=3, bias=True, padding=1)
        self.apply(init_xavier_fill_)

        def apply_bias(m):
            b = getattr(m, "bias", None)
            if b is None:
                return
            nn.init.constant_(b, thing_bias_value)

        self.thing_out.apply(apply_bias)

        self.in_channels = self.encoder.in_channels
        self.stuff_channels = stuff_channels
        self.thing_channels = thing_channels

    @override
    def forward(self, feat) -> Locations:
        x = self.encoder(feat)
        x = self.se(x)

        x_stuff = self.stuff_out(x)
        x_thing = self.thing_out(x)

        return Locations(
            stuff_map=x_stuff,
            thing_map=x_thing,
        )
