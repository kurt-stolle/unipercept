"""
This module implements functions for dealing with the state of the program.
The ccelerate library is adopted to handle distributed training and inference.
"""

import enum as E
import functools as F
import os
import sys
import threading
import typing as T

import accelerate.state
import accelerate.utils
import torch
import torch.distributed as dist
import torch.fx
import torch.types
import torch.utils.data
import typing_extensions as TX
from tensordict import TensorDict, TensorDictBase
from torch import Tensor
from torch.autograd import Function
from torch.futures import Future, collect_all, wait_all
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

import unipercept.log

__all__ = []


def get_state():
    return accelerate.state.PartialState()


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


@F.lru_cache
def get_interactive():
    from unipercept.config.env import get_env

    def default_interative_closure():
        if sys.stdin.isatty():  # Terminal
            return True
        if hasattr(sys, "ps1"):  # IPython, Python shell, etc.
            return True
        if sys.flags.interactive:  # Python launched with -i
            return True
        return os.isatty(sys.stdout.fileno())  # Redirected output

    return get_env(bool, "UP_INTERCTIVE", default=default_interative_closure)


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
    a_dist = auto_gather(a_dist).wait()
    assert isinstance(a_dist, Tensor), f"Expected Tensor, got {type(a_dist)}"
    # Compute total amount of samples
    a_total = int(a_dist.sum().item())

    a_off: list[int] = a_dist.cumsum(0).tolist()
    a_off = [0] + a_off[:-1]
    return a_total, a_off


##############################
# Wrappers around ccelerate #
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


def split_between_processes(
    inputs: T.Sequence | T.Mapping | torch.Tensor, apply_padding: bool = False
):
    return get_state().split_between_processes(inputs, apply_padding)


def void(*args: T.Any, **kwargs: T.Any) -> None:
    """
    function that does nothing.
    """


def noop(*args: T.Any, **kwargs: T.Any) -> T.Any:
    """
    function that does nothing and returns its input.
    """
    return args


########################
# Distributed handlers #
########################


class ReduceOp(E.StrEnum):
    SUM = E.auto()
    MEAN = E.auto()


class AutoGatherMode(E.StrEnum):
    CONCAT = E.auto()
    STACK = E.auto()


Destination: T.TypeAlias = int | T.Literal["all"]


class HandlerProtocol:
    __slots__ = ("__dict__", "__weakref__", "logger")

    process_index: int
    process_index_local: int
    process_count: int

    def auto_gather(
        self,
        item: Tensor,
        mode: AutoGatherMode | str,
        dst: Destination = "all",
        **kwargs,
    ) -> Future[Tensor]:
        raise NotImplementedError()

    def reduce_scatter(
        self,
        tgt,
        src,
        op=ReduceOp.SUM,
        **kwargs,
    ) -> Future[list[Tensor]]:
        raise NotImplementedError()

    def gather(
        self,
        tensor: Tensor,
        dst: Destination = "all",
        group=dist.group.WORLD,
        inplace=True,
        *args: Tensor,
    ) -> Future[Tensor]:
        raise NotImplementedError()

    def broadcast(
        self, tensor: Tensor, src: int, group=dist.group.WORLD, inplace=True
    ) -> Future[Tensor]:
        raise NotImplementedError()

    def reduce(
        self,
        tensor: Tensor,
        op: ReduceOp | str = ReduceOp.SUM,
        dst: Destination = "all",
        inplace=True,
        **kwargs,
    ) -> Future[Tensor]:
        raise NotImplementedError()

    # Internals
    _state_type: T.ClassVar[T.Callable[[], dict]]
    _sentinel: T.ClassVar[T.Self | None]

    def __new__(cls, *args, **kwargs):
        if cls is HandlerProtocol:
            msg = f"Cannot instantiate {cls.__name__} directly."
            raise TypeError(msg)
        if cls._sentinel is None:
            logger = unipercept.log.get_logger(f"{__name__} <{cls.__name__}>")
            h = cls._sentinel = super().__new__(cls)
            h.logger = logger
            h.__dict__ = cls._state_type()
            h.setup(*args, **kwargs)
        else:
            h = cls._sentinel
        assert h is not None
        return h

    def __init_subclass__(cls, *, state: type = dict) -> None:
        super().__init_subclass__()
        cls._state_type = state
        cls._sentinel = None

    def setup(cls, *args, **kwargs):
        raise NotImplementedError()


class _SingleProcessHandler(HandlerProtocol):
    r"""
    Non-distributed handler for single-process operation.
    """

    @TX.override
    def setup(self):
        self.process_index = self.process_index_local = 0
        self.process_count = 1

        self.logger.info(f"Using single-process state handler: {self}")

    @TX.override
    def auto_gather(
        self,
        item: Tensor,
        mode: AutoGatherMode | str,
        dst: Destination = "all",
        **kwargs,
    ) -> Future[Tensor]:
        tensor = torch.atleast_1d(item)
        if mode == AutoGatherMode.STACK:
            tensor = tensor.unsqueeze(0)
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        future = Future()
        future.set_result(tensor)
        return future

    @TX.override
    def reduce_scatter(
        self,
        tgt,
        src,
        op=ReduceOp.SUM,
        **kwargs,
    ) -> Future[list[Tensor]]:
        tgt[0] = src
        fut = Future()
        fut.set_result([tgt])
        return fut

    @TX.override
    def reduce(
        self,
        tensor: Tensor,
        op: ReduceOp | str = ReduceOp.SUM,
        dst: Destination = "all",
        inplace: bool = True,
        **kwargs,
    ) -> Future[Tensor]:
        if not inplace:
            tensor = tensor.clone()
        fut = Future()
        fut.set_result(tensor)

        return fut


class _DistributedHandler(HandlerProtocol):
    r"""
    Handler for ``torch.distributed``-based operations.
    """

    @TX.override
    def setup(self, *args, group=dist.group.WORLD, **kwargs):
        _xlr_state = accelerate.PartialState(*args, **kwargs)

        assert dist.is_initialized()
        self.group = group
        self.process_index = _xlr_state.process_index
        self.process_index_local = _xlr_state.local_process_index
        self.process_count = _xlr_state.num_processes

        self.logger.info(
            f"Using distributed state handler: {self} with backend {dist.get_backend()}"
        )

    @staticmethod
    def _maybe_to_future(work: T.Any) -> Future[Tensor]:
        assert work is not None
        if not isinstance(work, torch.futures.Future):

            def wrap_ddp_future(fut: Future[T.Sequence[Tensor]]) -> Future[Tensor]:
                res = fut.wait()
                assert len(res) == 1
                res_item = res[0]
                assert isinstance(res_item, Tensor), type(res_item)
                return res_item

            work = work.get_future().then(wrap_ddp_future)
        return work

    @TX.override
    def auto_gather(
        self,
        item: Tensor,
        mode: AutoGatherMode | str = AutoGatherMode.CONCAT,
        **kwargs,
    ) -> Future[Tensor]:
        def _gather_into_list(src: Tensor, mode: AutoGatherMode | str):
            work = dist.all_gather(
                [src.new_empty() for _ in range(self.process_count)],
                src,
                group=self.group,
                async_op=True,
            )
            work = self._maybe_to_future(work)

            if mode == AutoGatherMode.STACK:
                work = work.then(lambda f: torch.stack(f.wait(), dim=0))
            elif mode == AutoGatherMode.CONCAT:
                work = work.then(lambda f: torch.cat(f.wait(), dim=0))
            else:
                msg = f"Invalid auto gather mode: {mode}"
                raise ValueError(msg)

            return work

        def _gather_into_tensor(src: Tensor, mode: AutoGatherMode | str):
            if mode == AutoGatherMode.CONCAT:
                d_cat, *d_other = src.shape
                d_cat *= get_process_count()
                shape = (d_cat, *d_other)
            elif mode == AutoGatherMode.STACK:
                shape = (get_process_count(), *src.shape)
            else:
                msg = f"Invalid auto gather mode: {mode}"
                raise ValueError(msg)
            tgt = src.new_empty(shape)
            work = dist.all_gather_into_tensor(
                tgt, src, group=self.group, async_op=True
            )
            assert work is not None
            work = self._maybe_to_future(work)
            return work

        src = torch.atleast_1d(item)
        if not src.is_contiguous():
            src = src.contiguous()
        if dist.get_backend() == dist.Backend.GLOO:
            fn = _gather_into_list
        else:
            fn = _gather_into_tensor
        return fn(src, mode)

    @staticmethod
    def _convert_reduce_op(op: ReduceOp | str) -> dist.ReduceOp:
        if op == ReduceOp.SUM:
            return dist.ReduceOp.SUM
        if op == ReduceOp.MEAN:
            return dist.ReduceOp.AVG
        msg = f"Invalid reduce operation: {op}"
        raise ValueError(msg)

    @TX.override
    def reduce_scatter(
        self,
        tgt,
        src,
        op=ReduceOp.SUM,
        **kwargs,
    ) -> Future[list[Tensor]]:
        dist_op = self._convert_reduce_op(op)
        rank = self.process_index
        if tgt is None:
            tgt = src[rank]
        if tgt.dim() == 0:
            tgt = tgt.view(-1)
        tgt[:] = src[rank]
        ops: list[Future] = []
        for i in range(dist.get_world_size(self.group)):
            if i == rank:
                tmp = dist.reduce(tgt, rank, dist_op, self.group, async_op=True)
            else:
                tmp = dist.reduce(src[i], i, dist_op, self.group, async_op=True)
            assert tmp is not None
            tmp = self._maybe_to_future(tmp)
            ops.append(tmp)
        return collect_all(ops).then(wait_all)

    @TX.override
    def reduce(
        self,
        tgt: Tensor,
        op: ReduceOp | str = ReduceOp.SUM,
        dst: Destination = "all",
        inplace: bool = True,
        **kwargs,
    ) -> Future[Tensor]:
        dist_op = self._convert_reduce_op(op)
        if not inplace:
            tgt = tgt.clone()
        if dst == "all":
            work = dist.all_reduce(tgt, dist_op, self.group, async_op=True)
        else:
            work = dist.reduce(tgt, dst, dist_op, self.group, async_op=True)
        return self._maybe_to_future(work)


class _XLAState(threading.local):
    r"""
    The XLA state object, which we require to be thread-local.
    """

    def __init__(self, thread_local: bool = False):
        self._storage = {}

    def __get__(self, obj, objtype=None):
        return self._storage

    def __set__(self, obj, value):
        self._storage = value


class _XLAHandler(HandlerProtocol, state=_XLAState):
    r"""
    Handler for XLA-based operations, using the ``torch_xla`` library.
    """

    def setup(self):
        raise NotImplementedError("XLHandler is not implemented yet.")


def get_handler() -> HandlerProtocol:
    if get_state().distributed_type == accelerate.utils.DistributedType.XLA:
        h = _XLAHandler()
    elif (
        get_state().distributed_type
        in accelerate.utils.TORCH_DISTRIBUTED_OPERATION_TYPES
    ):
        h = _DistributedHandler()
    else:
        h = _SingleProcessHandler()
    assert h is not None
    return h


###############
# DDP helpers #
###############


if T.TYPE_CHECKING:
    _N = T.TypeVar("_N", bound=Tensor | dict[T.Any, Tensor] | T.Sequence[Tensor])

    def pad_across_processes(
        tensor: _N, dim: int = 0, pad_index: int = 0, pad_first: int = 0
    ) -> _N: ...

else:
    pad_across_processes = accelerate.utils.pad_across_processes


def reduce(
    tgt: torch.Tensor,
    *args,
    **kwargs,
) -> Future[Tensor]:
    return get_handler().reduce(tgt, *args, **kwargs)


def gather_object(objects: T.Any, dst: Destination = "all") -> T.Any:
    assert dst == "all", "Only all-gather is supported."
    return accelerate.utils.gather_object(objects)


def auto_gather(
    tgt: Tensor,
    mode: AutoGatherMode | str = AutoGatherMode.CONCAT,
    dst: Destination = "all",
) -> Future[Tensor]:
    return get_handler().auto_gather(tgt, mode, dst=dst)


@TX.deprecated("Use `tree_auto_gather` instead.")
def gather_tensordict(td: TensorDictBase) -> TensorDictBase:
    """
    Pads a TensorDict across processes and gathers it on the main process.
    """
    # Get the amount of batch dimensions, as this is lost during gathering
    batch_dims = td.batch_dims

    # Convert to dict
    td_dict: dict[str, Tensor] = td.to_dict()
    td_dict = pad_across_processes(td_dict)  # type: ignore
    td_dict = tree_auto_gather(td_dict)  # type: ignore

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


def _async_tree_unflatten(tree_futures: Future[list[Future[Tensor]]], spec: TreeSpec):
    tree = wait_all(tree_futures.wait())
    return tree_unflatten(tree, spec)


def tree_auto_gather(
    item: tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor] | Tensor,
    mode: AutoGatherMode = AutoGatherMode.CONCAT,
    **kwargs,
) -> Future[Tensor] | Future[list[Tensor]] | Future[dict[str, Tensor]]:
    r"""
    Gather a tree of tensors across all processes. If the input is a tensor, then it
    will be put into a list and gathered according to ``mode``.
    """
    handler = get_handler()
    if isinstance(item, Tensor):
        return handler.auto_gather(item, **kwargs)
    tree, spec = tree_flatten(item)
    work = []
    for item in tree:
        assert isinstance(item, Tensor)
        work.append(handler.auto_gather(item, mode, *kwargs, handler=handler))

    return collect_all(work).then(F.partial(_async_tree_unflatten, spec=spec))


torch.fx.wrap("tree_auto_gather")


def tree_reduce(
    item: tuple[Tensor, ...] | list[Tensor] | dict[str, Tensor] | Tensor,
    op: ReduceOp = ReduceOp.SUM,
    **kwargs,
) -> Future[Tensor] | Future[list[Tensor]] | Future[dict[str, Tensor]] | Tensor:
    r"""
    Reduce a tree of tensors across all processes.
    """
    handler = get_handler()
    if isinstance(item, Tensor):
        return handler.reduce(item, op, **kwargs)
    tree, spec = tree_flatten(item)
    work = []
    for item in tree:
        assert isinstance(item, Tensor)
        work.append(handler.reduce(item, op, **kwargs, handler=handler))

    return collect_all(work).then(F.partial(_async_tree_unflatten, spec=spec))


torch.fx.wrap("tree_reduce")


################
# Autograd DDP #
################


class ChainFunction(Function):
    @staticmethod
    @TX.override
    def forward(ctx, src, zero_grad, *tgt):
        ctx.save_for_backward(src)
        ctx.zero_grad = zero_grad
        return tgt

    @staticmethod
    @TX.override
    def backward(ctx, *grad_outputs):
        (tensor_to_consume,) = ctx.saved_tensors
        if ctx.zero_grad:
            fake_grad = torch.zeros_like(tensor_to_consume)
        else:
            fake_grad = torch.ones_like(tensor_to_consume)

        return (fake_grad, None) + grad_outputs


def chain_with_grad(src: Tensor, tgt: list[Tensor], zero_grad: bool = False):
    return ChainFunction.apply(src, zero_grad, *tgt)


torch.fx.wrap("chain_with_grad")


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


torch.fx.wrap("send_with_grad")


class RecvFunction(Function):
    @staticmethod
    @TX.override
    def forward(
        ctx, tgt, src: Tensor | None = None, group=dist.group.WORLD, tag=0, inplace=True
    ):
        if not inplace:
            tgt = torch.zeros_like(tgt).requires_grad_(False)
        ctx.src = src
        ctx.group = group
        ctx.tag = tag
        sender = dist.recv(tgt, src, group, tag)
        if src:
            assert sender == src
        else:
            ctx.src = sender
        sender = torch.tensor(sender)
        ctx.mark_non_differentiable(sender)
        return tgt, sender

    @staticmethod
    @TX.override
    def backward(ctx, grad_tensor, grad_sender):
        dist.send(grad_tensor, ctx.src, ctx.group, ctx.tag)
        return grad_tensor, None, None, None, None


def recv_with_grad(tgt: Tensor, src: Tensor, group=dist.group.WORLD, tag=0):
    """
    Variant of ``torch.distributed.recv`` that supports gradients.
    """
    return RecvFunction.apply(tgt, src, group, tag)


torch.fx.wrap("recv_with_grad")


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
        elif not inplace:
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


torch.fx.wrap("broadcast_with_grad")


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
        dist.scatter(grad_input, [], src=ctx.dst, group=ctx.group)
        return grad_input, None, None, None, None


def gather_with_grad(
    tgt: list[torch.Tensor],
    src: torch.Tensor,
    group=dist.group.WORLD,
    inplace=True,
    to: Destination = "all",
):
    """
    Variant of ``torch.distributed.gather`` that supports gradients.
    """
    if to == "all":
        return AllGatherFunction.apply(src, group, inplace, *tgt)
    return GatherFunction.apply(src, tgt, group, inplace)


torch.fx.wrap("gather_with_grad")


def auto_gather_with_grad(
    src: Tensor,
    mode: AutoGatherMode | str = AutoGatherMode.CONCAT,
    dst: Destination = "all",
) -> Tensor:
    """
    Variant of :func:`auto_gather` that supports gradients.
    """
    if not dist.is_initialized():
        return src if mode == AutoGatherMode.CONCAT else src.unsqueeze(0)
    tgt = [torch.zeros_like(src) for _ in range(dist.get_world_size())]
    gather_with_grad(tgt, src, dist.group.WORLD, True, dst)
    if mode == AutoGatherMode.CONCAT:
        return torch.cat(tgt, dim=0)
    if mode == AutoGatherMode.STACK:
        return torch.stack(tgt, dim=0)
    msg = f"Invalid auto gather mode: {mode}"
    raise ValueError(msg)


def tree_auto_gather_with_grad(
    src: Tensor,
    mode: AutoGatherMode = AutoGatherMode.CONCAT,
    group=dist.group.WORLD,
    inplace=True,
    dst: Destination = "all",
):
    """
    Variant of :func:`tree_auto_gather` that supports gradients.
    """
    assert all == "all", all
    src = torch.atleast_1d(src)
    if not src.is_contiguous():
        src = src.contiguous()
    if not check_distributed():
        return src if mode == AutoGatherMode.CONCAT else src.unsqueeze(0)
    tgt = [torch.zeros_like(src) for _ in range(dist.get_world_size(group))]
    tgt_items = AllGatherFunction.apply(src, group, inplace, *tgt)
    if not isinstance(tgt_items, list):
        tgt_items = [tgt_items]
    else:
        tgt_items = list(tgt_items)
    if mode == AutoGatherMode.CONCAT:
        return torch.cat(tgt_items, dim=0)
    if mode == AutoGatherMode.STACK:
        return torch.stack(tgt_items, dim=0)
    msg = f"Invalid auto gather mode: {mode}"
    raise ValueError(msg)


torch.fx.wrap("tree_auto_gather_with_grad")


class ScatterFunction(Function):
    @staticmethod
    @TX.override
    def forward(
        ctx,
        tgt: Tensor,
        src: int,
        group=dist.group.WORLD,
        inplace=True,
        *scatter_list,
    ):
        ctx.src = src
        ctx.group = group
        if not inplace:
            tgt = torch.zeros_like(tgt)
        if dist.get_rank(group) == src:
            ctx.save_for_backward(*scatter_list)
            scatter_list = list(scatter_list)
            dist.scatter(tgt, scatter_list, src=src, group=group)
        else:
            dist.scatter(tgt, [], src=src, group=group)
        return tgt

    @staticmethod
    @TX.override
    def backward(ctx, grad_tensor):
        if dist.get_rank(ctx.group) == ctx.src:
            grad_outputs = [torch.zeros_like(g) for g in ctx.saved_tensors]
            dist.gather(grad_tensor, grad_outputs, ctx.src, group=ctx.group)
            return (grad_tensor, None, None, None) + tuple(grad_outputs)
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


torch.fx.wrap("scatter_with_grad")
