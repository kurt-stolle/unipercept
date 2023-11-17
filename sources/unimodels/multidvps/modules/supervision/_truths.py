from __future__ import annotations

from typing import Mapping

import torch
from tensordict import TensorDictBase
from torch import Tensor
from unicore.utils.tensorclass import Tensorclass

from ._assign_stuff import Stuff
from ._assign_things import Things


class Truths(Tensorclass):
    thing: TensorDictBase
    stuff: TensorDictBase
    semmap: Tensor
    insmap: Tensor

    def __post_init__(self, *args, **kwargs):
        assert len(self.thing) == len(self.stuff), f"{len(self.thing)} != {len(self.stuff)}"

    @torch.no_grad()
    def type_as_(self, other: Tensor):
        self.semmap = self.semmap.type_as(other)
        self.insmap = self.insmap.type_as(other)

        def type_as_if_floating(v: Tensor) -> Tensor:
            if v.is_floating_point():
                return v.type_as(other)
            else:
                return v

        v = self.thing.apply(type_as_if_floating, inplace=False)
        self.thing = v

        v = self.stuff.apply(type_as_if_floating, inplace=False)
        self.stuff = v

    @torch.no_grad()
    def mask_instances(self, thing_nums: Mapping[str, int]) -> tuple[Tensor, Tensor, Tensor, int]:
        masks = torch.cat(
            [gt.instances.insts[:, : thing_nums[i], ...] for i, gt in self.thing.items()],
            dim=1,
        )
        indices = torch.cat(
            [gt.instances.indices_mask[:, : thing_nums[i]] for i, gt in self.thing.items()],
            dim=1,
        )
        labels: Tensor = torch.cat(
            [
                torch.stack([gt.instances.categories[:, : thing_nums[i]], gt.instances.ids[:, : thing_nums[i]]], dim=-1)
                for i, gt in self.thing.items()
            ],
            dim=1,
        )
        indices = indices.type(torch.bool)

        return masks, indices, labels, int(indices.sum())

    @torch.no_grad()
    def mask_semantic(self, stuff_nums: Mapping[str, int]) -> tuple[Tensor, Tensor, int]:
        masks = torch.cat(
            [gt.labels[:, : stuff_nums[i], ...] for i, gt in self.stuff.items()],
            dim=1,
        )
        indices = torch.cat(
            [gt.indices[:, : stuff_nums[i]] for i, gt in self.stuff.items()],
            dim=1,
        )
        indices = indices.type(torch.bool)

        return masks, indices, int(indices.sum())
