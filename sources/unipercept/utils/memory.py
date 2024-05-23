"""

See Also
--------
- https://github.com/facebookresearch/detectron2/blob/bce6d7262b1065498481be1d6708c8dbb142975a/detectron2/utils/memory.py

"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import wraps

import torch

__all__ = ["retry_if_cuda_oom"]


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores CUDA OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "CUDA out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func, cpu_ok: bool = True):
    """
    Makes a function retry itself after encountering
    pytorch's CUDA OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to CPUs.
    In this case, it expects the function to dispatch to CPU implementation.
    The return values may become CPU tensors as well and it's user's
    responsibility to convert it back to CUDA tensor if needed.

    Parameters
    ----------
    func
        A stateless callable that takes tensor-like objects as arguments

    Returns
    -------
        A callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Note:
        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cpu(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cpu")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)
        torch.cuda.empty_cache()
        if cpu_ok:
            with _ignore_torch_cuda_oom():
                return func(*args, **kwargs)
            new_args = (maybe_to_cpu(x) for x in args)
            new_kwargs = {k: maybe_to_cpu(v) for k, v in kwargs.items()}
            return func(*new_args, **new_kwargs)
        return func(*args, **kwargs)

    return wrapped
