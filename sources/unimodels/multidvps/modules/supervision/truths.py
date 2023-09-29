from __future__ import annotations

from typing import Mapping

import torch
from tensordict import TensorDictBase
from torch import Tensor
from unicore.utils.tensorclass import Tensorclass

from .assign_stuff import Stuff
from .assign_things import Things


class Truths(Tensorclass):
    thing: TensorDictBase[str, Things]
    stuff: TensorDictBase[int, Stuff]
    semmap: Tensor
    insmap: Tensor

    def __post_init__(self, *args, **kwargs):
        assert len(self.thing) == len(self.stuff), f"{len(self.thing)} != {len(self.stuff)}"

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

    def mask_instances(self, thing_nums: Mapping[str, int]) -> tuple[Tensor, Tensor, int]:
        masks = torch.cat(
            [gt.instances.insts[:, : thing_nums[i], ...] for i, gt in self.thing.items()],
            dim=1,
        )
        indices = torch.cat(
            [gt.instances.indices_mask[:, : thing_nums[i]] for i, gt in self.thing.items()],
            dim=1,
        )
        indices = indices.type(torch.bool)

        return masks, indices, int(indices.sum())

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
