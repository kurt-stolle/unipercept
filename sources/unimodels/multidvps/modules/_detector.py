"""Implements the multi-scale detector module."""

from __future__ import annotations

import typing as T

import tensordict
import torch
import torch.nn as nn
from typing_extensions import override
from unicore.utils.tensorclass import Tensorclass

__all__ = ["Detection", "Detector"]

if T.TYPE_CHECKING:
    from tensordict.tensordict import TensorDict, TensorDictBase
    from torch import Size, Tensor

    from ._encoder import Encoder
    from ._kernels import Kernelizer
    from ._position import Localizer, Locations


class Detection(Tensorclass):
    thing_map: torch.Tensor
    stuff_map: torch.Tensor
    kernel_spaces: tensordict.TensorDict

    @property
    def hw(self) -> Size:
        return self.thing_map.shape[-2:]


class Detector(nn.Module):
    def __init__(self, in_features: T.Sequence[str], kernelizer: Kernelizer, localizer: Localizer):
        super().__init__()

        self.in_features = in_features
        self.kernelizer = kernelizer
        self.localizer = localizer

    @property
    def keys(self):
        return self.kernelizer.keys

    @override
    def forward(self, feats: TensorDictBase) -> TensorDict:
        return tensordict.TensorDict(
            {key: self._detect(feats.get(key)) for key in self.in_features},
            batch_size=feats.batch_size,
            device=feats.device,
        )

    def _detect(self, feat: Tensor) -> Detection:
        loc = T.cast(Locations, self.localizer(feat))
        k_spaces = T.cast(tensordict.TensorDict, self.kernelizer(feat))

        hw = loc.thing_map.shape[-2:]
        assert hw == loc.stuff_map.shape[-2:], (hw, loc.stuff_map.shape)
        assert all(
            hw == k_space.shape[-2:] for k_space in k_spaces.values()
        ), f"Expected kernel spaces to have shape {hw}, got {[k_space.shape for k_space in k_spaces.values()]}!"

        return Detection(
            loc.thing_map,
            loc.stuff_map,
            k_spaces,
            batch_size=feat.shape[:1],
            device=feat.device,
        )
