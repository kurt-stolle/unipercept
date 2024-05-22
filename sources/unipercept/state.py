"""
This module implements functions for dealing with the state of the program.
The Accelerate library is adopted to handle distributed training and inference.
"""

from __future__ import annotations

import dataclasses as D
import functools as F
import os
import sys
import typing as T

import accelerate.utils
import torch
import torch.distributed
import torch.types
import torch.utils.data
from tensordict import TensorDict, TensorDictBase

__all__ = []
_xlr_state = accelerate.PartialState()

####################
# Unipercept state #
####################


@F.lru_cache()
def get_interactive():
    from unipercept.config import get_env

    def default_interative_closure():
        if sys.stdin.isatty():  # Terminal
            return True
        if hasattr(sys, "ps1"):  # IPython, Python shell, etc.
            return True
        if sys.flags.interactive:  # Python launched with -i
            return True
        return os.isatty(sys.stdout.fileno())  # Redirected output

    return get_env(bool, "UP_INTERACTIVE", default=default_interative_closure)


##################################
# Data and multiprocessing utils #
##################################


def get_total_batchsize(
    dataloader: torch.utils.data.DataLoader,
    device: torch.types.Device,
) -> tuple[int, list[int]]:
    a = len(dataloader)
    # Gather the size of dataloaders across all processes
    a_dist = torch.tensor([a], dtype=torch.int64, device=device)
    a_dist = gather(a_dist)
    assert isinstance(a_dist, torch.Tensor), f"Expected Tensor, got {type(a_dist)}"
    # Compute total amount of samples
    a_total = int(a_dist.sum().item())

    a_off: list[int] = a_dist.cumsum(0).tolist()
    a_off = [0] + a_off[:-1]
    return a_total, a_off


##############################
# Wrappers around Accelerate #
##############################


def get_process_index(local=False):
    return _xlr_state.local_process_index if local else _xlr_state.process_index


def get_process_count() -> int:
    return _xlr_state.num_processes


def check_main_process(local=False):
    return _xlr_state.is_local_main_process if local else _xlr_state.is_main_process


def check_debug_enabled():
    return _xlr_state.debug


def barrier(msg: str | None = None):
    return _xlr_state.wait_for_everyone()


def main_process_first(local: bool = False):
    if local:
        return _xlr_state.local_main_process_first()
    else:
        return _xlr_state.main_process_first()


print = _xlr_state.print


def check_distributed() -> bool:
    return _xlr_state.use_distributed


def on_process():
    return _xlr_state.on_process


def on_last_process():
    return _xlr_state.on_last_process


def on_main_process():
    return _xlr_state.on_main_process


###############
# DDP helpers #
###############

if T.TYPE_CHECKING:
    _N = T.TypeVar(
        "_N", bound=torch.Tensor | dict[T.Any, torch.Tensor] | T.Sequence[torch.Tensor]
    )

    def gather(tensor: _N) -> _N: ...

    def pad_across_processes(
        tensor: _N, dim: int = 0, pad_index: int = 0, pad_first: int = 0
    ) -> _N: ...

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


def cpus_available():
    from multiprocessing import cpu_count

    try:
        return len(os.sched_getaffinity(0))
    except:
        pass
    return max(cpu_count(), 1)
