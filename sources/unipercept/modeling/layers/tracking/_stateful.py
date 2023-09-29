from __future__ import annotations

import copy
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import unitrack
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from typing_extensions import override

__all__ = ["StatefulTracker"]


class _MemoryReader(nn.Module):
    """
    Wrapper around a tracklet memory that enables reading from it in the forward
    pass. This is necessary because the memory is a stateful module that is not
    compatible with the functional API.
    """

    def __init__(self, memory: unitrack.TrackletMemory, **kwargs):
        super().__init__(**kwargs)
        self.memory = memory

    @override
    def forward(self, frame: int):
        return self.memory.read(frame)


class _MemoryWriter(nn.Module):
    """
    Wrapper around a tracklet memory that enables writing to it in the forward
    pass. This is necessary because the memory is a stateful module that is not
    compatible with the functional API.
    """

    def __init__(self, memory: unitrack.TrackletMemory, **kwargs):
        super().__init__(**kwargs)
        self.memory = memory

    @override
    def forward(self, ctx: TensorDictBase, obs: TensorDictBase, new: TensorDictBase):
        return self.memory.write(ctx, obs, new)


def _split_persistent_buffers(
    module: nn.Module, prefix: str = ""
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Split the buffers of a module into two dictionaries: one for
    buffers that are shared across sequences (persistent) and one for buffers that are
    unique to every sequence (non-persistent).
    """

    shared = {}
    unique = {}

    for name, buf in module.named_buffers(prefix=prefix, recurse=False):
        if buf in module._non_persistent_buffers_set:
            unique[name] = buf
        else:
            shared[name] = buf

    for name, submodule in module.named_children():
        s, u = _split_persistent_buffers(submodule, prefix=f"{prefix}.{name}")
        shared.update(s)
        unique.update(u)

    return shared, unique


class StatefulTracker(nn.Module):
    """
    A wrapper around Unitrack tracker and tracklets that enables
    stateful tracking of objects for every separate sequence.
    """

    mem_buffers: Mapping[str | int, dict[str, torch.Tensor]]
    mem_params: dict[str, torch.Tensor]

    def __init__(self, tracker: unitrack.MultiStageTracker, memory: unitrack.TrackletMemory):
        super().__init__()

        self.tracker = tracker
        self.memory = memory

    @cached_property
    def memory_reader(self) -> _MemoryReader:
        return _MemoryReader(memory=self.memory).to("meta")

    @cached_property
    def memory_writer(self) -> _MemoryWriter:
        return _MemoryWriter(memory=self.memory).to("meta")

    @cached_property
    def memory_storage(self) -> tuple[dict[str, torch.nn.Parameter], dict[str, torch.Tensor], defaultdict]:
        prefix = "memory"
        params = dict(self.memory.named_parameters(prefix=prefix))
        buffers_shared, buffers_unique = _split_persistent_buffers(self.memory, prefix=prefix)
        buffers_unique_map = defaultdict(lambda: copy.deepcopy(buffers_unique))

        return params, buffers_shared, buffers_unique_map

    @override
    def forward(self, ctx: TensorDictBase, things: TensorDictBase, key: str | int, frame: int) -> Tensor:
        params, buffers_shared, buffers_unique = self.memory_storage
        pbd = (params, {**buffers_shared, **buffers_unique[key]})

        # Read
        state_ctx, state_obs = torch.func.functional_call(self.memory_reader, pbd, (frame), strict=True)

        # Step
        inp = TensorDict({**ctx.to_dict(), "things": things.to_dict()}, batch_size=[])

        state_obs, new = self.tracker(state_ctx, state_obs, inp)

        # Write
        ids = torch.func.functional_call(self.memory_writer, pbd, (state_ctx, state_obs, new), strict=True)

        return ids
