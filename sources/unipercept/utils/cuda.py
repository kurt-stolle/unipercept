"""
Implements utility functions that assert whether the current system supports certain features.
"""


from __future__ import annotations

__all__ = []


def check_cuda_gpus_available(min: int = 1, max: int | None = None) -> bool:
    """
    Checks if the current system has at least `min` GPUs available and at most `max` GPUs available. If `max` is not
    specified, then it is assumed that there is no upper bound on the number of GPUs available.
    """
    import torch.cuda

    if not torch.cuda.is_available():
        return False

    device_count = torch.cuda.device_count()
    if device_count < min:
        return False
    if max is not None and device_count > max:
        return False
    return True


P2P_IB_UNSUPPORTED_DEVICES = ["RTX 3090", "RTX 40"]


def has_p2pib_support():
    """
    Checks support for P2P and IB communications.
    """
    from torch.cuda import device_count, get_device_name, is_available

    if is_available() and device_count() > 0:
        device_name = get_device_name()
        return not any(n in device_name for n in P2P_IB_UNSUPPORTED_DEVICES)
    else:
        return True
