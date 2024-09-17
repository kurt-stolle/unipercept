"""

See Also
--------
- https://github.com/facebookresearch/detectron2/blob/bce6d7262b1065498481be1d6708c8dbb142975a/detectron2/utils/memory.py

"""

from __future__ import annotations

import contextlib
import functools
import gc
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


def release_memory():
    """
    Attempt to free memory from the GPU, XPU, NPU, or MPS.
    """
    gc.collect()
    # torch.xpu.empty_cache()
    # torch.mlu.empty_cache()
    # torch.npu.empty_cache()
    # torch.mps.empty_cache()
    torch.cuda.empty_cache()


_OOM_EXCEPTION_PATTERNS = [
    "CUDA out of memory.",
    "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",
    "DefaultCPUAllocator: can't allocate memory",
]


def check_oom_exception(e: Exception) -> bool:
    """
    Checks whether an exception is CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory.
    """
    if isinstance(e, RuntimeError) and len(e.args) == 1:
        return any(err in e.args[0] for err in _OOM_EXCEPTION_PATTERNS)
    return False


_R = T.TypeVar("_R")
_P = T.ParamSpec("_P")


def find_executable(
    fn: T.Callable[T.Concatenate[int, _P], _R] | None = None, /, *, max_iter: int = 10
) -> (
    T.Callable[T.Concatenate[_P], _R]
    | T.Callable[[T.Callable[T.Concatenate[_P], _R]], T.Callable[T.Concatenate[_P], _R]]
):
    """
    A basic decorator that will try to execute `function`. If it fails from exceptions
    related to out-of-memory or CUDNN, the function will be retried with the first
    integer argument increased by one.

    This will continue until the function executes successfully or the user interrupts
    the process.

    Parameters
    ----------
    fn :
        The function to decorate. If not provided, the decorator will return a partial
        function that can be called with the function to decorate.
    max_iterations : int | None
        The maximum number of iterations to attempt before raising an error.

    Returns
    -------
    Callable[[int, ...], ...] :
        The decorated function or a partial function that can be called with the function to
        decorate.

    Raises
    ------
    StopIteration :
        If the maximum number of iterations is reached.

    ```
    """
    if fn is None:
        return functools.partial(find_executable_batch_size, max_iter=max_iter)  # type: ignore

    assert max_iter is None or max_iter >= 1, f"{max_iter=} <= 1"

    n = 0

    @functools.wraps(fn)
    def _fn_wrap(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        nonlocal n
        release_memory()
        while True:
            if n >= max_iter:
                raise StopIteration("Max iterations reached.")
            try:
                return fn(n, *args, **kwargs)
            except Exception as e:
                if check_oom_exception(e):
                    n += 1
                    release_memory()
                    continue
                raise

    return _fn_wrap  # type: ignore        return x
