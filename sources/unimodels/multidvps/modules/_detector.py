"""Implements the multi-scale detector module."""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override
from unicore.utils.tensorclass import Tensorclass

__all__ = ["Detection", "Detector"]

if T.TYPE_CHECKING:
    from ._kernels import Kernelizer
    from ._position import Localizer


class Detection(Tensorclass):
    thing_map: torch.Tensor
    stuff_map: torch.Tensor
    kernel_spaces: TensorDict

    @property
    def hw(self) -> torch.Size:
        return self.thing_map.shape[-2:]


class Detector(nn.Module):
    in_features: T.Final[T.List[str]]

    def __init__(self, in_features: T.Sequence[str], kernelizer: Kernelizer, localizer: Localizer):
        super().__init__()

        self.in_features = list(in_features)
        self.kernelizer = kernelizer
        self.localizer = localizer

    @property
    def keys(self):
        return self.kernelizer.keys

    @override
    def forward(self, feats: T.Dict[str, torch.Tensor]) -> T.Dict[str, Detection]:
        return {key: self._detect(feats.get(key)) for key in self.in_features}

    def _detect(self, feat: torch.Tensor) -> Detection:
        loc = self.localizer(feat)
        k_spaces = T.cast(TensorDict, self.kernelizer(feat))

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
