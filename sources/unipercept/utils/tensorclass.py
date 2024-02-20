"""
Utility to create a `tensordict`-like tensorclass using a superclass instead of
the provided decorator `@tensorclass`.

This is useful when you want typing to work properly, and is more explicit.
"""

from __future__ import annotations

import dataclasses as D
import types
import typing as T

import torch
import torch.distributed
import torch.types
import torch.utils._pytree as pytree
import typing_extensions as TX
from tensordict import TensorDict, TensorDictBase, tensorclass

if T.TYPE_CHECKING:
    from tensordict.base import IndexType as IndexType
    from tensordict.tensordict import NestedKey as NestedKey

from typing_extensions import override

__all__ = ["Tensorclass", "TensorDict", "TensorDictBase"]
_CompatibleType: T.TypeAlias = torch.Tensor


@T.dataclass_transform()
class _TensorclassMeta(type):
    """
    TensorclassMeta is a metaclass that wraps the `@tensorclass` decorator around the child.
    """

    def __new__(
        metacls, name: str, bases: tuple[type, ...], ns: dict[str, T.Any], **kwds
    ):
        # if len(bases) == 0:
        #     return super().__new__(cls, name, tuple(bases), ns, **kwds)

        bases = types.resolve_bases(bases)
        tc = super().__new__(metacls, name, tuple(bases), ns, **kwds)

        return tensorclass(tc)


@T.dataclass_transform()
class Tensorclass(metaclass=_TensorclassMeta):
    """
    Tensorclass is a class that allows you to create a `tensordict`-like
    tensorclass using a superclass instead of the provided decorator `@tensorclass`.
    """

    def __post_init__(self):
        pass

    def _flatten(self) -> T.Tuple[T.List[T.Any], pytree.Context]:
        values = []
        values += list(self._tensordict.values())  # type: ignore
        values += list(self._non_tensordict.values())  # type: ignore

        keys = []
        keys += list(self._tensordict.keys())
        keys += list(self._non_tensordict.keys())

        context = {
            "keys": keys,
            "batch_size": self.batch_size,
            # "names": self.names,
        }

        return values, context

    @classmethod
    def stack(cls, *others: T.Self) -> T.Self:
        """
        Stacks multiple tensorclasses together.
        """
        if len(others) == 0:
            raise ValueError("Must provide at least one tensorclass to stack.")

        if len(others) == 1:
            return others[0]

        td = T.cast(TensorDictBase, torch.stack(others))  # type: ignore
        return cls.from_tensordict(td)

    @classmethod
    def _unflatten(cls, values: T.List[T.Any], context: pytree.Context) -> T.Self:
        obj = cls(
            **dict(zip(context["keys"], values)),
            batch_size=context["batch_size"],
            # names=context["names"],
        )
        return obj

    def __getattr__(self, name: str) -> T.Any:
        try:
            return super().__getattr__(name)  # type: ignore
        except AttributeError:
            return self.get(name)  # type: ignore

    @TX.override
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        pytree.register_pytree_node(cls, cls._flatten, cls._unflatten)

    # Type hints for methods and properties added by the @tensorclass decorator
    if T.TYPE_CHECKING:
        _: D.KW_ONLY
        batch_size: torch.Size | T.Sequence[int]
        device: T.Optional[torch.types.Device | str | None] = None

        # Tensorclass-specific methods
        @classmethod
        def from_tensordict(
            cls,
            tensordict: TensorDictBase,
            non_tensordict: dict[str, T.Any] | None = None,
        ) -> T.Self:
            ...

        # Inherited methods
        @property
        def shape(self) -> torch.Size:
            ...

        @property
        def names(self):
            ...

        @names.setter
        def names(self, value):
            ...

        def refine_names(self, *names):
            ...

        def rename(self, *names, **rename_map):
            ...

        def rename_(self, *names, **rename_map):
            ...

        def size(self, dim: int | None = None) -> torch.Size | int:
            ...

        @property
        def requires_grad(self) -> bool:
            ...

        def ndimension(self) -> int:
            ...

        @property
        def ndim(self) -> int:
            ...

        def dim(self) -> int:
            ...

        def clear_device_(self) -> T.Self:
            ...

        def is_shared(self) -> bool:
            ...

        def state_dict(self) -> dict[str, T.Any]:
            ...

        def load_state_dict(self, state_dict: dict[str, T.Any]) -> T.Self:
            ...

        def is_memmap(self) -> bool:
            ...

        def numel(self) -> int:
            ...

        def send(self, dst: int, init_tag: int = 0, pseudo_rand: bool = False) -> None:
            ...

        def recv(self, src: int, init_tag: int = 0, pseudo_rand: bool = False) -> int:
            ...

        def isend(self, dst: int, init_tag: int = 0, pseudo_rand: bool = False) -> int:
            ...

        def irecv(self) -> tuple[int, list[torch.Future]] | list[torch.Future] | None:
            ...

        def reduce(
            self,
            dst,
            op=torch.distributed.ReduceOp.SUM,
            async_op=False,
            return_premature=False,
        ):
            ...

        def pop(self) -> _CompatibleType:
            ...

        def apply_(self, fn: T.Callable) -> T.Self:
            ...

        def apply(self) -> T.Self:
            ...

        def as_tensor(self):
            ...

        def update(self) -> T.Self:
            ...

        def update_(self) -> T.Self:
            ...

        def update_at_(self) -> T.Self:
            ...

        def items(self) -> T.Iterator[tuple[str, _CompatibleType]]:
            ...

        def values(self) -> T.Iterator[_CompatibleType]:
            ...

        @property
        def sorted_keys(self) -> list[NestedKey]:
            ...

        def flatten(self, start_dim=0, end_dim=-1):
            ...

        def unflatten(self, dim, unflattened_size):
            ...

        def exclude(self, *keys: str, inplace: bool = False) -> T.Self:
            ...

        def copy_(self, tensordict: T.Self) -> T.Self:
            ...

        def copy_at_(self, tensordict: T.Self, idx: IndexType) -> T.Self:
            ...

        def get_at(self) -> _CompatibleType:
            ...

        def memmap_like(self, prefix: str | None = None) -> T.Self:
            ...

        def detach(self) -> T.Self:
            ...

        def to_h5(self):
            ...

        def to_tensordict(self):
            ...

        def zero_(self) -> T.Self:
            ...

        def unbind(self, dim: int) -> tuple[T.Self, ...]:
            ...

        def chunk(self, chunks: int, dim: int = 0) -> tuple[T.Self, ...]:
            ...

        def clone(self, recurse: bool = True) -> T.Self:
            ...

        def cuda(self, device: int = 0) -> T.Self:
            ...

        def masked_select(self, mask: torch.Tensor) -> T.Self:
            ...

        def to_dict(self) -> dict[str, T.Any]:
            ...

        def unsqueeze(self, dim: int) -> T.Self:
            ...

        def squeeze(self, dim: int | None = None) -> T.Self:
            ...

        def reshape(self) -> T.Self:
            ...

        def split(self, split_size: int | list[int], dim: int = 0) -> list[T.Self]:
            ...

        def gather(self) -> T.Self:
            ...

        def view(self) -> T.Self:
            ...

        def permute(self) -> T.Self:
            ...

        def all(self, dim: int | None = None) -> bool | T.Self:
            ...

        def any(self, dim: int | None = None) -> bool | T.Self:
            ...

        def get_sub_tensordict(self, idx: IndexType) -> T.Self:
            ...

        def flatten_keys(self) -> T.Self:
            ...

        def unflatten_keys(self) -> T.Self:
            ...

        def fill_(self, key: str, value: float | bool) -> T.Self:
            ...

        def empty(self) -> T.Self:
            ...

        def is_empty(self) -> bool:
            ...

        @property
        def is_locked(self) -> bool:
            ...

        @is_locked.setter
        def is_locked(self, value: bool) -> None:
            ...

        def lock_(self) -> T.Self:
            ...

        def unlock_(self) -> T.Self:
            ...

        @property
        def batch_dims(self) -> int:
            ...

        def pin_memory(self) -> T.Self:
            ...

        def expand(self, *shape: int) -> T.Self:
            ...

        def set(self) -> T.Self:
            ...

        def set_(self) -> T.Self:
            ...

        def del_(self, key: str) -> T.Self:
            ...

        def rename_key_(self) -> T.Self:
            ...

        def entry_class(self, key: NestedKey) -> type:
            ...

        def set_at_(self) -> T.Self:
            ...

        def get(self) -> T.Any:
            ...

        def share_memory_(self) -> T.Self:
            ...

        def detach_(self) -> T.Self:
            ...

        def memmap_(self) -> T.Self:
            ...

        @classmethod
        def load_memmap(cls, prefix: str) -> T.Self:
            ...

        def to(
            self, dest: torch.types.Device | torch.Size | type, **kwargs: T.Any
        ) -> T.Self:
            ...

        def masked_fill_(self, mask: torch.Tensor, value: float | int | bool) -> T.Self:
            ...

        def masked_fill(self, mask: torch.Tensor, value: float | bool) -> T.Self:
            ...

        def is_contiguous(self) -> bool:
            ...

        def contiguous(self) -> T.Self:
            ...

        def select(self) -> T.Self:
            ...

        def keys(self) -> T.Sequence[str]:
            ...
