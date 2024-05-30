"""
This module implements functions for dealing with the state of the program.
The Accelerate library is adopted to handle distributed training and inference.
"""

from __future__ import annotations
import functools as F
import threading
import enum as E
from torch import Tensor
from torch.futures import Future, wait_all, collect_all
import types
import abc
import os
import sys
import typing as T
import typing_extensions as TX
import accelerate.utils
import accelerate.state
import torch
import torch.distributed as dist
import torch.types
import torch.utils.data
from tensordict import TensorDict, TensorDictBase, is_tensor_collection
from unipercept.utils.tensorclass import Tensorclass
from torch.autograd import Function
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
import torch

__all__ = []


_XLR_STATE: accelerate.PartialState | None = None


def get_state():
    global _XLR_STATE

    if _XLR_STATE is None:
        _XLR_STATE = accelerate.state.PartialState()
    return _XLR_STATE


###################
# Multiprocessing #
###################


def cpus_available():
    r"""
    Returns the number of CPUs available to the current process.

    This is preferred over `os.cpu_count()` as it is more reliable in HPC environments.
    """
    from multiprocessing import cpu_count

    try:
        return len(os.sched_getaffinity(0))
    except:
        pass
    return max(cpu_count(), 1)


class WorkList(list[dist.Work | Future]):
    r"""
    Wrapper class around the result object for distributed operations that have
    ``async_op=True``.

    This class is used to wait for a collective of async operations
    to complete.
    """

    def wait(self):
        for op in self:
            op.wait()

    def is_completed(self):
        for op in self:
            if isinstance(op, dist.Work):
                if not op.is_completed():
                    return False
                continue
            if isinstance(op, Future):
                if not op.done():
                    return False
                continue
            msg = f"Expected Work or Future, got {type(op)}"
            raise TypeError(msg)
        return True


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
    assert isinstance(a_dist, Tensor), f"Expected Tensor, got {type(a_dist)}"
    # Compute total amount of samples
    a_total = int(a_dist.sum().item())

    a_off: list[int] = a_dist.cumsum(0).tolist()
    a_off = [0] + a_off[:-1]
    return a_total, a_off


##############################
# Wrappers around Accelerate #
##############################


def get_process_index(local=False):
    return get_state().local_process_index if local else get_state().process_index


def get_device():
    return get_state().device


def get_process_count() -> int:
    return get_state().num_processes


def check_main_process(local=False):
    return get_state().is_local_main_process if local else get_state().is_main_process


def check_debug_enabled():
    return get_state().debug


def barrier():
    return get_state().wait_for_everyone()


def main_process_first(local: bool = False):
    if local:
        return get_state().local_main_process_first()
    else:
        return get_state().main_process_first()


def print(*args, **kwargs):
    get_state().print(*args, **kwargs)


def check_distributed() -> bool:
    return get_state().use_distributed


def check_distributed_gpu() -> bool:
    return get_state().use_distributed_gpu


def on_process(index: int):
    return F.partial(get_state().on_process, process_index=index)


def on_last_process():
    return get_state().on_last_process


def on_main_process():
    return get_state().on_main_process


def void(*args: T.Any, **kwargs: T.Any) -> None:
    """
    A function that does nothing.
    """
    pass


def noop(*args: T.Any, **kwargs: T.Any) -> T.Any:
    """
    A function that does nothing and returns its input.
    """
    return args


########################
# Distributed handlers #
########################


class AutoGatherMode(E.StrEnum):
    CONCAT = E.auto()
    STACK = E.auto()


class HandlerProtocol:
    __slots__ = ("__init__", "__dict__", "__weakref__")

    process_index: int
    process_index_local: int
    process_count: int

    def all_auto_gather(
        self, item: Tensor, mode: AutoGatherMode | str, **kwargs
    ) -> Future[Tensor]:
        raise NotImplementedError()

    def reduce_scatter(
        self,
        tensor: Tensor,
        tensor_list: list[Tensor],
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        **kwargs,
    ) -> Future[list[Tensor]]:
        raise NotImplementedError()

    # Internals
    _state_type: T.ClassVar[T.Callable[[], dict]]
    _sentinel: T.ClassVar[HandlerProtocol | None]

    def __new__(cls):
        if cls is HandlerProtocol:
            msg = f"Cannot instantiate {cls.__name__} directly."
            raise TypeError(msg)
        if cls._sentinel is None:
            h = cls._sentinel = super().__new__(cls)
            setattr(h, "__dict__", h._state_type())
            setattr(h, "__init__", cls.__init__)
        else:
            h = cls._sentinel
            setattr(h, "__init__", cls.__init__)
        return h

    def reset(self):
        self.__dict__.clear()
        self.__class__.__init__(self)

    def __init_subclass__(cls, *, state: type = dict) -> None:
        super().__init_subclass__()
        cls._state_type = state
        cls._sentinel = None


class AccelerateHandler(HandlerProtocol):
    """
    From v4.0.0, we use the Accelerate library to handle distributed training and inference.

    This class is a wrapper that uses the ``PartialState`` class to do the heavy lifting
    for us.
    """

    def __init__(self, *args, **kwargs):
        self._xlr = accelerate.PartialState(*args, **kwargs)

        self.process_index = self._xlr.process_index
        self.process_index_local = self._xlr.local_process_index
        self.process_count = self._xlr.num_processes


class SingleProcessHandler(HandlerProtocol):
    def __init__(self):
        super().__init__()

        self.process_index = self.process_index_local = 0
        self.process_count = 1

    @TX.override
    def all_auto_gather(
        self, item: Tensor, mode: AutoGatherMode | str, **kwargs
    ) -> Future[Tensor]:
        tensor = torch.atleast_1d(item)
        if mode == AutoGatherMode.STACK:
            tensor = tensor.unsqueeze(0)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        future = Future()
        future.set_result(tensor)
        return future


class DistributedHandler(HandlerProtocol):
    def __init__(self):
        super().__init__()

        assert dist.is_initialized()

    @TX.override
    def all_auto_gather(
        self,
        item: Tensor,
        mode: AutoGatherMode | str = AutoGatherMode.CONCAT,
        *,
        group=dist.group.WORLD,
        **kwargs,
    ) -> Future[Tensor]:
        src = torch.atleast_1d(item)
        if not src.is_contiguous():
            src = src.contiguous()
        if dist.get_backend() == dist.Backend.GLOO:
            work = dist.all_gather(
                [src.new_empty() for _ in range(self.process_count)],
                src,
                group=group,
                async_op=True,
            )
            assert work is not None
            if not isinstance(work, torch.futures.Future):
                work = work.get_future()
            if mode == AutoGatherMode.STACK:
                work = work.then(lambda f: torch.stack(f.wait(), dim=0))
            elif mode == AutoGatherMode.CONCAT:
                work = work.then(lambda f: torch.cat(f.wait(), dim=0))
            else:
                msg = f"Invalid auto gather mode: {mode}"
                raise ValueError(msg)
            return work

        if mode == AutoGatherMode.CONCAT:
            d_cat, *d_other = src.shape
            d_cat *= get_process_count()
            shape = (d_cat, *d_other)
        elif mode == AutoGatherMode.STACK:
            shape = (get_process_count(), *src.shape)
        else:
            msg = f"Invalid auto gather mode: {mode}"
            raise ValueError(msg)
        dst = src.new_empty(shape)
        work = dist.all_gather_into_tensor(dst, src, group=group, async_op=True)
        assert work is not None
        if not isinstance(work, torch.futures.Future):
            work = work.get_future()
        return work.then(lambda f: torch.cat(f.wait(), dim=0))

    @TX.override
    def reduce_scatter(
        self,
        dst,
        src,
        op=dist.ReduceOp.SUM,
        *,
        group=dist.group.WORLD,
        **kwargs,
    ) -> Future[list[Tensor]]:
        rank = self.process_index
        if dst is None:
            dst = src[rank]
        if dst.dim() == 0:
            dst = dst.view(-1)
        dst[:] = src[rank]
        ops: list[Future] = []
        for i in range(dist.get_world_size(group)):
            if i == rank:
                tmp = dist.reduce(dst, rank, op, group, async_op=True)
            else:
                tmp = dist.reduce(src[i], i, op, group, async_op=True)
            assert tmp is not None
            if not isinstance(tmp, Future):
                tmp = tmp.get_future()
            ops.append(tmp)
        return collect_all(ops).then(wait_all)


class XLAState(threading.local):
    def __init__(self, thread_local: bool = False):
        self._storage = {}

    def __get__(self, obj, objtype=None):
        return self._storage

    def __set__(self, obj, value):
        self._storage = value


class XLAHandler(HandlerProtocol, state=XLAState):
    def __init__(self):
        super().__init__()

        raise NotImplementedError("XLAHandler is not implemented yet.")


def get_handler() -> HandlerProtocol:
    if get_state().distributed_type == accelerate.utils.DistributedType.XLA:
        return XLAHandler
    elif get_state().distributed_type in accelerate.utils.dist_OPERATION_TYPES:
        return DistributedHandler
    else:
        return SingleProcessHandler


###############
# DDP helpers #
###############

if T.TYPE_CHECKING:
    _N = T.TypeVar("_N", bound=Tensor | dict[T.Any, Tensor] | T.Sequence[Tensor])

    def gather(tensor: _N) -> _N: ...

    def pad_across_processes(
        tensor: _N, dim: int = 0, pad_index: int = 0, pad_first: int = 0
    ) -> _N: ...

else:
    gather = accelerate.utils.gather
    pad_across_processes = accelerate.utils.pad_across_processes


@TX.deprecated("Use `all_auto_gather` instead.")
def gather_tensordict(td: TensorDictBase) -> TensorDictBase:
    """
    Pads a TensorDict across processes and gathers it on the main process.
    """
    # Get the amount of batch dimensions, as this is lost during gathering
    batch_dims = td.batch_dims

    # Convert to dict
    td_dict: dict[str, Tensor] = td.to_dict()
    td_dict = pad_across_processes(td_dict)  # type: ignore
    td_dict = gather(td_dict)  # type: ignore

    # Recover TensorDict object
    td = TensorDict.from_dict(td_dict)
    td.batch_size = td.batch_size[:batch_dims]

    if not isinstance(td, TensorDict):
        fn = next(
            iter(
                (
                    getattr(td, "from_tensordict", None),
                    getattr(td, "_from_tensordict", None),
                )
            ),
            None,
        )
        if fn is not None:
            td = fn(td_dict)

    return td


def tree_unflatten_async(tree_futures: Future[list[Future[Tensor]]], spec: TreeSpec):
    tree = wait_all(tree_futures.wait())
    return tree_unflatten(tree, spec)


def all_auto_gather(
    item: tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor] | Tensor,
    mode: AutoGatherMode = AutoGatherMode.CONCAT,
    *,
    handler: HandlerProtocol | None = None,
    **kwargs,
) -> Future[Tensor] | Future[list[Tensor]] | Future[dict[str, Tensor]] | Tensor:
    if handler is None:
        handler = get_handler()
    if isinstance(item, Tensor):
        return handler.all_auto_gather(item, **kwargs)

    tree, spec = tree_flatten(item)
    work = []
    for item in tree:
        assert isinstance(item, Tensor)
        work.append(handler.all_auto_gather(item, mode, *kwargs, handler=handler))

    return collect_all(work).then(F.partial(tree_unflatten_async, spec=spec))


################
# Autograd DDP #
################


class ChainFunction(Function):
    @staticmethod
    @TX.override
    def forward(ctx, src, zero_grad, *dst):
        ctx.save_for_backward(src)
        ctx.zero_grad = zero_grad
        return dst

    @staticmethod
    @TX.override
    def backward(ctx, *grad_outputs):
        (tensor_to_consume,) = ctx.saved_tensors
        if ctx.zero_grad:
            fake_grad = torch.zeros_like(tensor_to_consume)
        else:
            fake_grad = torch.ones_like(tensor_to_consume)

        return (fake_grad, None) + grad_outputs


def chain_with_grad(src: Tensor, dst: T.List[Tensor], zero_grad: bool = False):
    return ChainFunction.apply(src, zero_grad, *dst)


class SendFunction(Function):
    @staticmethod
    @TX.override
    def forward(ctx, src, dst, group=dist.group.WORLD, tag=0):
        ctx.save_for_backward(src)
        ctx.dst = dst
        ctx.group = group
        ctx.tag = tag
        dist.send(src, dst, group, tag)
        return src.new_tensor([])

    @staticmethod
    @TX.override
    def backward(ctx, grad_output):
        (tensor,) = ctx.saved_tensors
        grad_tensor = torch.zeros_like(tensor)
        dist.recv(grad_tensor, ctx.dst, ctx.group, ctx.tag)

        return grad_tensor, None, None, None


def send_with_grad(src: Tensor, dst: Tensor, group=dist.group.WORLD, tag=0):
    """
    Variant of ``torch.distributed.send`` that supports gradients.
    """
    return SendFunction.apply(src, dst, group, tag)


class RecvFunction(Function):
    @staticmethod
    @TX.override
    def forward(
        ctx, dst, src: Tensor | None = None, group=dist.group.WORLD, tag=0, inplace=True
    ):
        if not inplace:
            dst = torch.zeros_like(dst).requires_grad_(False)
        ctx.src = src
        ctx.group = group
        ctx.tag = tag
        sender = dist.recv(dst, src, group, tag)
        if src:
            assert sender == src
        else:
            ctx.src = sender
        sender = torch.tensor(sender)
        ctx.mark_non_differentiable(sender)
        return dst, sender

    @staticmethod
    @TX.override
    def backward(ctx, grad_tensor, grad_sender):
        dist.send(grad_tensor, ctx.src, ctx.group, ctx.tag)
        return grad_tensor, None, None, None, None


def recv_with_grad(dst: Tensor, src: Tensor, group=dist.group.WORLD, tag=0):
    """
    Variant of ``torch.distributed.recv`` that supports gradients.
    """
    return RecvFunction.apply(dst, src, group, tag)


class BroadcastFunction(Function):
    @staticmethod
    @TX.override
    def forward(ctx, tensor, src, group=dist.group.WORLD, inplace=True):
        ctx.src = src
        ctx.group = group
        if dist.get_rank(group) == src:
            if not inplace:
                with torch.no_grad():
                    tensor = tensor.clone().requires_grad_(False)
        else:
            if not inplace:
                tensor = torch.zeros_like(tensor).requires_grad_(False)
        dist.broadcast(tensor, src, group)
        return tensor

    @staticmethod
    @TX.override
    def backward(ctx, grad_output):
        dist.reduce(grad_output, ctx.src, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None, None, None


def broadcast_with_grad(tensor: Tensor, src: int, group=dist.group.WORLD, inplace=True):
    """
    Variant of ``torch.distributed.broadcast`` that supports gradients.
    """
    return BroadcastFunction.apply(tensor, src, group, inplace)


class AllGatherFunction(Function):
    @staticmethod
    @TX.override
    def forward(ctx, tensor, group, inplace, *args):
        ctx.save_for_backward(tensor)
        ctx.group = group
        gather_list = list(args)
        if not inplace:
            gather_list = [torch.zeros_like(g) for g in gather_list]
        dist.all_gather(gather_list, tensor, group)
        return tuple(gather_list)

    @staticmethod
    @TX.override
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        get_handler().reduce_scatter(grad_out, list(grads), group=ctx.group).wait()
        return (grad_out, None, None) + grads


def all_gather_with_grad(
    dst: list[torch.Tensor], src: Tensor, group=dist.group.WORLD, inplace=True
):
    """
    Variant of ``torch.distributed.all_gather`` that supports gradients.
    """
    if not check_distributed():
        dst[0] = src
        return dst
    return AllGatherFunction.apply(src, group, inplace, *dst)


def all_auto_gather_with_grad(
    src: Tensor,
    mode: AutoGatherMode = AutoGatherMode.CONCAT,
    group=dist.group.WORLD,
    inplace=True,
):
    """
    Variant of :func:`all_auto_gather` that supports gradients.
    """
    src = torch.atleast_1d(src)
    if not src.is_contiguous():
        src = src.contiguous()
    if not check_distributed():
        return src if mode == AutoGatherMode.CONCAT else src.unsqueeze(0)
    dst = [torch.zeros_like(src) for _ in range(dist.get_world_size(group))]
    dst_items = AllGatherFunction.apply(src, group, inplace, *dst)
    if not isinstance(dst_items, tuple):
        dst_items = [dst_items]
    else:
        dst_items = list(dst_items)
    if mode == AutoGatherMode.CONCAT:
        return torch.cat(dst_items, dim=0)
    elif mode == AutoGatherMode.STACK:
        return torch.stack(dst_items, dim=0)
    else:
        msg = f"Invalid auto gather mode: {mode}"
        raise ValueError(msg)


class GatherFunction(Function):
    @staticmethod
    @TX.override
    def forward(ctx, src: Tensor, dst: int, group: T.Any, inplace: bool, *args: Tensor):
        ctx.dst = dst
        ctx.group = group
        ctx.save_for_backward(src)
        if dist.get_rank(group) == dst:
            gather_list = list(args)
            if not inplace:
                gather_list = [torch.zeros_like(g) for g in gather_list]
            dist.gather(src, gather_list=gather_list, dst=dst, group=group)
            return tuple(gather_list)
        else:
            dist.gather(src, [], dst=dst, group=group)
            return src.new_tensor([])

    @staticmethod
    @TX.override
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        if dist.get_rank(ctx.group) == ctx.dst:
            grad_outputs = list(grads)
            dist.scatter(grad_input, grad_outputs, src=ctx.dst, group=ctx.group)
            return (grad_input, None, None, None) + grads
        else:
            dist.scatter(grad_input, [], src=ctx.dst, group=ctx.group)
            return grad_input, None, None, None, None


def gather_with_grad(
    dst: list[torch.Tensor],
    src: torch.Tensor,
    group=dist.group.WORLD,
    inplace=True,
):
    """
    Variant of ``torch.distributed.gather`` that supports gradients.
    """
    return GatherFunction.apply(src, dst, group, inplace)


class ScatterFunction(Function):
    @staticmethod
    @TX.override
    def forward(
        ctx,
        dst: Tensor,
        src: int,
        group=dist.group.WORLD,
        inplace=True,
        *scatter_list,
    ):
        ctx.src = src
        ctx.group = group
        if not inplace:
            dst = torch.zeros_like(dst)
        if dist.get_rank(group) == src:
            ctx.save_for_backward(*scatter_list)
            scatter_list = list(scatter_list)
            dist.scatter(dst, scatter_list, src=src, group=group)
        else:
            dist.scatter(dst, [], src=src, group=group)
        return dst

    @staticmethod
    @TX.override
    def backward(ctx, grad_tensor):
        if dist.get_rank(ctx.group) == ctx.src:
            grad_outputs = [torch.zeros_like(g) for g in ctx.saved_tensors]
            dist.gather(grad_tensor, grad_outputs, ctx.src, group=ctx.group)
            return (grad_tensor, None, None, None) + tuple(grad_outputs)
        else:
            dist.gather(grad_tensor, [], ctx.src, group=ctx.group)
            return grad_tensor, None, None, None, None


def scatter_with_grad(
    tensor: torch.Tensor,
    src: int,
    group=dist.group.WORLD,
    inplace=True,
    *scatter_list,
):
    """
    Variant of ``torch.distributed.scatter`` that supports gradients.
    """
    return ScatterFunction.apply(tensor, src, group, inplace, *scatter_list)
