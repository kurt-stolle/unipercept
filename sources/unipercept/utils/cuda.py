"""
Implements utility functions that assert whether the current system supports certain features.
"""

from __future__ import annotations

import typing as T

import torch.cuda

__all__ = []


def check_cuda_gpus_available(min: int = 1, max: int | None = None) -> bool:
    """
    Checks if the current system has at least `min` GPUs available and at most `max` GPUs available. If `max` is not
    specified, then it is assumed that there is no upper bound on the number of GPUs available.
    """
    if not torch.cuda.is_available():
        return False
    device_count = torch.cuda.device_count()
    if device_count < min:
        return False
    if max is not None and device_count > max:
        return False
    return True


P2P_IB_BLACKLIST: T.Collection[str] = {"RTX 3090", "RTX 40"}


def has_p2pib_support():
    """
    Checks support for P2P and IB communications.
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.cuda.get_device_name() not in P2P_IB_BLACKLIST
    return True


SHARED_MEMORY_SIZE_MAP: T.Mapping[tuple[int, int], int] = {
    (8, 0): 163000,
    (8, 6): 99000,
    (8, 7): 163000,
    (8, 9): 99000,
    (9, 0): 227000,
    (7, 5): 64000,
    (7, 0): 96000,
}


def get_shm_size():
    """
    Lookup the shared memory size for the current CUDA device using the capability
    level of the device.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        msg = "No CUDA device context available."
        raise RuntimeError(msg)
    cap = (
        torch.cuda.get_device_properties(0).major,
        torch.cuda.get_device_properties(0).minor,
    )
    if cap not in SHARED_MEMORY_SIZE_MAP:
        msg = (
            f"Missing entry in shared memory size map for CUDA capability: {cap}. "
            "You may set UP_CUDA_OVERRIDE_SHM_SIZE to avoid this error by providing your "
            f" own value, or add an entry to the mapping in {__file__!r} (and "
            " submit a pull request)."
        )
        raise KeyError(msg)
    return SHARED_MEMORY_SIZE_MAP[cap]
