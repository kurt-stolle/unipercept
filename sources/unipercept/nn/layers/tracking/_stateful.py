from __future__ import annotations

import copy
from collections import defaultdict
from functools import cached_property
from typing import Mapping

import torch
import torch.nn as nn
import unitrack
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from typing_extensions import override

__all__ = ["StatefulTracker"]


# class _MemoryReader(nn.Module):
#     """
#     Wrapper around a tracklet memory that enables reading from it in the forward
#     pass. This is necessary because the memory is a stateful module that is not
#     compatible with the functional API.
#     """

#     def __init__(self, memory: unitrack.TrackletMemory, **kwargs):
#         super().__init__(**kwargs)
#         self.memory = memory

#     @override
#     def forward(self, frame: int):
#         return self.memory.read(frame)


# class _MemoryWriter(nn.Module):
#     """
#     Wrapper around a tracklet memory that enables writing to it in the forward
#     pass. This is necessary because the memory is a stateful module that is not
#     compatible with the functional API.
#     """

#     def __init__(self, memory: unitrack.TrackletMemory, **kwargs):
#         super().__init__(**kwargs)
#         self.memory = memory

#     @override
#     def forward(self, ctx: TensorDictBase, obs: TensorDictBase, new: TensorDictBase):
#         return self.memory.write(ctx, obs, new)


class _MemoryReadWriter(nn.Module):
    """
    Wrapper around a tracklet memory that enables writing and reading in the forward pass.
    This is necessary because the memory is a stateful module that is not compatible with the functional API.
    """

    def __init__(self, memory: unitrack.TrackletMemory):
        super().__init__()
        self.memory = memory

    @override
    def forward(
        self,
        write: bool,
        transaction: int | tuple[TensorDictBase, TensorDictBase, TensorDictBase],
    ):
        if write:
            ctx, obs, new = transaction
            return self.memory.write(ctx, obs, new)
        else:
            (frame,) = transaction
            return self.memory.read(frame)


def _split_persistent_buffers(
    module: nn.Module, prefix: str
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Split the buffers of a module into two dictionaries: one for
    buffers that are shared across sequences (persistent) and one for buffers that are
    unique to every sequence (non-persistent).
    """

    shared = {}
    unique = {}

    for name, buf in module.named_buffers(prefix="", recurse=False):
        name_prefixed = f"{prefix}.{name}"
        if name in module._non_persistent_buffers_set:
            unique[name_prefixed] = buf
        else:
            shared[name_prefixed] = buf

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

    def __init__(
        self, tracker: unitrack.MultiStageTracker, memory: unitrack.TrackletMemory
    ):
        super().__init__()

        self.tracker = tracker
        self.memory_delegate = _MemoryReadWriter(memory)

    # @cached_property
    # def memory_reader(self) -> _MemoryReader:
    #     return _MemoryReader(memory=self.memory).to("meta")

    # @cached_property
    # def memory_writer(self) -> _MemoryWriter:
    #     return _MemoryWriter(memory=self.memory).to("meta")

    @cached_property
    def memory_storage(
        self,
    ) -> tuple[dict[str, torch.nn.Parameter], dict[str, torch.Tensor], defaultdict]:
        prefix = "memory"
        memory = self.memory_delegate.memory
        params = dict(memory.named_parameters(prefix=prefix))
        buffers_shared, buffers_unique = _split_persistent_buffers(
            memory, prefix=prefix
        )
        buffers_unique_map = defaultdict(lambda: copy.deepcopy(buffers_unique))

        return params, buffers_shared, buffers_unique_map

    @override
    def forward(self, x: TensorDict, n: int, key: int, frame: int) -> Tensor:
        """
        Parameters
        ----------
        x: TensorDictBase
            Represents the state of the current iteration.
        n: int
            The amount of detections in the current frame, amounts to the length of
            IDs returned
        key: int
            The key that identifies the sequence in which the detections have been made
        frame: int
            The current frame number, used to identify the temporal position within
            the sequence.

        Returns
        -------
        Tensor[n]
            Assigned instance IDs
        """
        # Read
        params, buffers_shared, buffers_unique = self.memory_storage
        buffers_unique = buffers_unique[key]
        pbd: dict[str, torch.Tensor] = {**params, **buffers_shared, **buffers_unique}
        state_ctx, state_obs = torch.func.functional_call(
            self.memory_delegate, pbd, (False, (frame,)), strict=True
        )

        # Step
        state_obs, new = self.tracker(state_ctx, state_obs, x, n)

        print(f"After tracking, got\n- observations: {state_obs}\n- new: {new}")

        # Write
        ids: torch.Tensor = torch.func.functional_call(
            self.memory_delegate, pbd, (True, (state_ctx, state_obs, new)), strict=True
        )

        # assert pbd["memory.states." + unitrack.constants.KEY_FRAME] == frame, (
        #     f"Frame number {frame} was not updated in the memory. "
        #     f"Found frame {pbd[unitrack.constants.KEY_FRAME]} in states!"
        # )

        for buf_key, buf_val in pbd.items():
            if buf_key in params:
                continue
            if buf_key in buffers_shared:
                buffers_shared[buf_key] = buf_val
                continue
            if buf_key in buffers_unique:
                print(
                    f"Assign: {buf_key} = {buf_val.tolist()}, was: {buffers_unique[buf_key].tolist()}"
                )
                buffers_unique[buf_key] = buf_val
                continue

            msg = (
                f"Buffer with key {buf_key!r} not found in parameters and buffers dict!"
            )
            raise KeyError(msg)

        print(f"The current buffers has memory address {id(buffers_unique)}")

        print(buffers_unique)

        return ids
