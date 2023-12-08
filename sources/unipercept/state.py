"""
This module implements functions for dealing with the state of the program.
The Accelerate library is adopted to handle distributed training and inference.
"""

from __future__ import annotations

import dataclasses as D
import typing as T

import accelerate.utils
import torch
from tensordict import TensorDict, TensorDictBase


@D.dataclass(kw_only=True, slots=True)
class _ProcessStateManager:
    interactive: bool = False


_state_backend = accelerate.PartialState()
_state_unipercept = _ProcessStateManager()


##############################
# Wrappers around Accelerate #
##############################


def get_interactive():
    return _state_unipercept.interactive


def get_process_index(local=False):
    return _state_backend.local_process_index if local else _state_backend.process_index


def get_process_count() -> int:
    return _state_backend.num_processes


def check_main_process(local=False):
    return _state_backend.is_local_main_process if local else _state_backend.is_main_process


def check_debug_enabled():
    return _state_backend.debug


def barrier(*args):
    return _state_backend.wait_for_everyone()


main_process_first = _state_backend.main_process_first
local_main_process_first = _state_backend.local_main_process_first
print = _state_backend.print


def on_process():
    return _state_backend.on_process


def on_last_process():
    return _state_backend.on_last_process


def on_main_process():
    return _state_backend.on_main_process


###############
# DDP helpers #
###############

if T.TYPE_CHECKING:
    _N = T.TypeVar("_N", bound=torch.Tensor | dict[T.Any, torch.Tensor] | T.Sequence[torch.Tensor])

    def gather(tensor: _N) -> _N:
        ...

    def pad_across_processes(tensor: _N, dim: int = 0, pad_index: int = 0, pad_first: int = 0) -> _N:
        ...

else:
    gather = accelerate.utils.gather
    pad_across_processes = accelerate.utils.pad_across_processes


def gather_tensordict(td: TensorDictBase) -> TensorDict:
    """
    Pads a TensorDict across processes and gathers it on the main process.
    """
    # Get the amount of batch dimensions, as this is lost during gathering
    batch_dims = td.batch_dims

    # Convert to dict
    td_dict: dict[str, torch.Tensor] = td.to_dict()
    td_dict = pad_across_processes(td_dict)  # type: ignore
    td_dict = gather(td_dict)  # type: ignore

    # Recover TensorDict object
    td = TensorDict.from_dict(td_dict)
    td.batch_size = td.batch_size[:batch_dims]

    return td