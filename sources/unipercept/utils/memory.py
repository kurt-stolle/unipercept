"""

See Also
--------
- https://github.com/facebookresearch/detectron2/blob/bce6d7262b1065498481be1d6708c8dbb142975a/detectron2/utils/memory.py

"""

from __future__ import annotations

import contextlib
import functools
import typing as T

import torch

CUDA_OOM_ERROR: T.Final[str] = "CUDA out of memory. "


@contextlib.contextmanager
def ignore_cuda_oom():
    """
    A context which ignores CUDA OOM errors.
    """
    try:
        yield
    except RuntimeError as e:
        if CUDA_OOM_ERROR in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(
    fn: T.Callable | None = None, *, cpu_ok: bool = True
) -> T.Callable:
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
    fn: callable or None
        A stateless callable that takes tensor-like objects as arguments. If None, then
        this function will return a decorator that can be used to wrap a function.

    Returns
    -------
        A callable which retries `func` if OOM is encountered.

    Examples:
    ::
        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on CPU even if inputs are on GPU

    Notes
    -----

        1. When converting inputs to CPU, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    if fn is None:
        return functools.partial(retry_if_cuda_oom, cpu_ok=cpu_ok)

    @functools.wraps(fn)
    def fn_with_retry(*args, **kwargs):
        with ignore_cuda_oom():
            return fn(*args, **kwargs)
        torch.cuda.empty_cache()
        if cpu_ok:
            with ignore_cuda_oom():
                return fn(*args, **kwargs)
            new_args = (_maybe_to_cpu(x) for x in args)
            new_kwargs = {k: _maybe_to_cpu(v) for k, v in kwargs.items()}
            return fn(*new_args, **new_kwargs)
        return fn(*args, **kwargs)

    return fn_with_retry


def _maybe_to_cpu(x):
    try:
        like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
    except AttributeError:
        like_gpu_tensor = False
    if like_gpu_tensor:
        return x.to(device="cpu")
    else:
        return x
