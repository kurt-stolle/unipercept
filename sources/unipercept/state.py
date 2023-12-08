"""
This module implements functions for dealing with the state of the program.
The Accelerate library is adopted to handle distributed training and inference.
"""

from __future__ import annotations

import dataclasses as D

import accelerate.utils
import torch
from tensordict import TensorDict, TensorDictBase


@D.dataclass(kw_only=True, slots=True)
class _ProcessStateManager:
    interactive: bool = False


_state_xlr = accelerate.PartialState()
_state_up = _ProcessStateManager()


def get_interactive():
    return _state_up.interactive


def get_process_index(local=False):
    return _state_xlr.local_process_index if local else _state_xlr.process_index


def get_process_count() -> int:
    return _state_xlr.num_processes


def check_main_process(local=False):
    return _state_xlr.is_local_main_process if local else _state_xlr.is_main_process


def check_debug_enabled():
    return _state_xlr.debug


def barrier(*args):
    return _state_xlr.wait_for_everyone()


main_process_first = _state_xlr.main_process_first
local_main_process_first = _state_xlr.local_main_process_first
print = _state_xlr.print


def on_process():
    return _state_xlr.on_process


def on_last_process():
    return _state_xlr.on_last_process


def on_main_process():
    return _state_xlr.on_main_process


def gather_tensordict(td: TensorDictBase) -> TensorDict:
    """
    Pads a TensorDict across processes and gathers it on the main process.
    """
    # Get the amount of batch dimensions, as this is lost during gathering
    batch_dims = td.batch_dims

    # Convert to dict
    td_dict: dict[str, torch.Tensor] = td.to_dict()
    td_dict = accelerate.utils.pad_across_processes(td_dict)  # type: ignore
    td_dict = accelerate.utils.gather(td_dict)  # type: ignore

    # Recover TensorDict object
    td = TensorDict.from_dict(td_dict)
    td.batch_size = td.batch_size[:batch_dims]

    return td
