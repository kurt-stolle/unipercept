from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase, tensorclass
from torch import Tensor
from typing_extensions import override


# type: ignore
class Stuff:
    labels: Tensor
    masks: Tensor
    indices: Tensor
    scoremap: Tensor

    @property
    def num_instances(self) -> int:
        return int(self.indices.sum(dim=-1).max())


class StuffAssigner(nn.Module):
    def __init__(
        self, *, with_things: bool, all_classes: bool, num_cats: int, ignore_val: int, embeddings: dict[int, int]
    ):
        super().__init__()

        self.num_cats = num_cats
        self.ignore_val = ignore_val

        self.with_things = with_things
        self.all_classes = all_classes

        self.embeddings = embeddings

    @override
    def forward(
        self,
        semmap: Tensor,
        *,
        hw_detections: Mapping[str, torch.Size],
        hw_embedding: torch.Size,
    ) -> TensorDictBase:
        semmap = self._parse_map(semmap, hw_embedding)

        return TensorDict.from_dict({key: self._assign(semmap, hw) for key, hw in hw_detections.items()})

    def _parse_map(self, semmap: Tensor, hw: torch.Size):
        semmap = semmap.masked_fill(semmap == self.ignore_val, self.num_cats)

        # One-hot encode the semantic map
        semmap = F.one_hot(semmap, num_classes=self.num_cats + 1)
        semmap = semmap.permute(0, 3, 1, 2).float()

        # Discards the ignore-class
        semmap = semmap[:, : self.num_cats, :, :]

        # Scale to output size
        assert semmap.ndim == 4, f"Expected 4D tensor, got {semmap.ndim}D tensor"

        semmap = F.interpolate(
            semmap,
            size=hw,
            mode="bilinear",
            align_corners=False,
        )

        return semmap

    def _assign(self, semmap: Tensor, hw: torch.Size) -> Stuff:
        regions = self._create_regions(semmap, hw)
        labels, masks, valid = self._create_instances(semmap, regions)

        # convert to dict
        return Stuff(
            labels=labels,
            masks=masks,
            indices=valid,
            scoremap=regions,
            batch_size=labels.shape[:2],
        )  # type: ignore

    def _create_regions(self, semmap: Tensor, hw: torch.Size) -> Tensor:
        assert semmap.ndim == 4, f"Expected 4D tensor, got {semmap.ndim}D tensor"

        regions = F.interpolate(
            semmap,
            size=hw,
            mode="bilinear",
            align_corners=False,
        )
        regions[regions < 0.5] = 0.0
        regions.clamp_(max=1.0)

        return regions

    def _create_instances(self, semmap: Tensor, scoremap: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        active = scoremap.sum(dim=(-2, -1)) > 0
        valid = torch.ones_like(active, dtype=torch.int).cumsum_(dim=-1) <= active.sum(dim=-1, keepdims=True)

        labels = torch.zeros_like(semmap)
        labels[valid] = semmap[active]

        mask = torch.zeros_like(scoremap, dtype=torch.bool)
        mask[valid] = scoremap[active].bool()

        return labels, mask, valid

    def _translate(self, semmap: Tensor):
        semmap_thing = torch.full_like(semmap, self.ignore_val)
        for e_from, e_to in self.embeddings.items():
            semmap_thing.masked_fill_(semmap == e_from, e_to)

        return semmap_thing
