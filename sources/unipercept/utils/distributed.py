"""
Wraps PyTorch distributed and adds some extra features
"""

import typing as T

import torch
import torch.distributed as D

__all__ = ["is_main_process", "wait_for_sync"]


def get_world_size() -> int:
    if not D.is_available():
        return 1
    if not D.is_initialized():
        return 1
    return D.get_world_size()


def get_rank() -> int:
    if not D.is_available():
        return 0
    if not D.is_initialized():
        return 0
    return D.get_rank()


def is_main_process() -> bool:
    """
    Check whether the distributed backend was initialized and we are the main process.
    """
    return get_rank() == 0


def wait_for_sync() -> None:
    """
    Wait for all processes to synchronize.
    """

    if not D.is_available() or not D.is_initialized() or D.get_world_size() <= 1:
        return  # no waiting required
    if D.get_backend() == D.Backend.NCCL:
        D.barrier(device_ids=[torch.cuda.current_device()])
    else:
        D.barrier()


def distributed_concat(
    tensor: T.Tuple[torch.Tensor] | T.List[torch.Tensor] | T.Mapping[str, torch.Tensor] | torch.Tensor,
    num_total_examples: T.Optional[int] = None,
) -> T.Any:
    try:
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(distributed_concat(t, num_total_examples) for t in tensor)
        if isinstance(tensor, T.Mapping):
            return type(tensor)({k: distributed_concat(t, num_total_examples) for k, t in tensor.items()})
        tensor = torch.atleast_1d(tensor).contiguous()
        output_tensors = [tensor.clone() for _ in range(D.get_world_size())]
        D.all_gather(output_tensors, tensor)

        # TODO: async?
        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        if num_total_examples is not None:
            concat = concat[:num_total_examples]
        return concat
    except AssertionError as err:
        raise AssertionError("Not currently using distributed training") from err
