"""
Implements a handler for writing results to a file from multiple processes.
"""

from __future__ import annotations

import abc
import collections
import concurrent.futures
import functools
import gc
import itertools as I
import json
import os
import shutil
import sys
import typing as T

import torch
import torch.multiprocessing as M
import torch.types
import typing_extensions as TX
from tensordict import (
    LazyStackedTensorDict,
    MemoryMappedTensor,
    PersistentTensorDict,
    TensorDict,
    TensorDictBase,
)
from tensordict.utils import TensorDictFuture
from torch import Tensor

from unipercept import file_io
from unipercept.log import get_logger
from unipercept.state import (
    barrier,
    check_main_process,
    cpus_available,
    gather,
    gather_tensordict,
    get_process_count,
    main_process_first,
    on_main_process,
)
from unipercept.utils.tensorclass import Tensorclass
from unipercept.utils.typings import Pathable

_logger = get_logger(__name__)

_NO_DEFAULT: T.Final = T.cast(torch.Tensor, "_no_default_")
_IndexType: T.TypeAlias = torch.Tensor | slice | T.Iterable[int]


class MemoryMappedTensorMeta(T.TypedDict):
    """
    Metadata for tensors saved with every tensordict in a memory-mapped directory.
    """

    dtype: torch.dtype
    shape: torch.Size


class LazyStackedMemmapTensorView:
    """
    Represents a tensor within a sub-view of a stacked memory-mapped tensordict
    """

    __slots__ = ("dtype", "shape", "_executor", "_path", "_key", "_view")

    def __init__(
        self,
        path: str,
        key: str,
        view: _IndexType,
        dtype: torch.dtype,
        shape: torch.Size,
        executor: concurrent.futures.Executor,
    ):
        from tensordict.utils import _STRDTYPE2DTYPE

        self._path: T.Final = path
        self._key: T.Final = key
        self._view: T.Final = self._view_to_indices(view)
        self.dtype: T.Final = (
            _STRDTYPE2DTYPE[dtype] if isinstance(dtype, str) else dtype
        )
        self.shape: T.Final = torch.Size(shape)
        self._executor: T.Final = executor

    @staticmethod
    def _view_to_indices(view: _IndexType) -> list[int]:
        if isinstance(view, slice):
            assert view.start is not None
            assert view.stop is not None
            return list(range(view.start, view.stop, view.step or 1))
        if isinstance(view, torch.Tensor):
            return view.tolist()
        elif isinstance(view, T.Iterable):
            return list(view)
        msg = f"Unsupported sub-view type, got: {view} ({type(view)})"
        raise TypeError(msg)

    def __len__(self) -> int:
        return len(self._view)

    @TX.override
    def __getstate__(self):
        msg = f"{self.__class__.__name__} does not support pickling"
        raise NotImplementedError(msg)

    def __setstate__(self, state):
        msg = f"{self.__class__.__name__} does not support pickling"
        raise NotImplementedError(msg)

    def __getitem__(self, index: int | _IndexType) -> Tensor:
        if isinstance(index, int):
            return self._load_at(index)
        if isinstance(index, slice):
            parent_indices = self._view[index]
        else:
            parent_indices = [self._view[i] for i in index]
        futures = [self._executor.submit(self._load_at, i) for i in parent_indices]
        tensors = [f.result() for f in futures]
        return torch.stack(T.cast(list[torch.Tensor], tensors))

    def _load_at(self, i: int) -> Tensor:
        path = file_io.Path(self._path) / str(i) / f"{self._key}.memmap"
        return (
            torch.from_file(
                str(path),
                shared=False,
                dtype=self.dtype,
                size=self.shape.numel(),
                requires_grad=False,
            )
            .reshape(self.shape)
            .pin_memory()
        )

        # return tensor.clone(memory_format=torch.contiguous_format)

    def to_tensor(self) -> Tensor:
        return self[:]


class LazyStackedMemmapTensorDict(TensorDictBase):
    """
    This tensordict is a read-only view of a dictory with containig previously
    memory-mapped tensordicts.
    """

    _is_locked = True
    _is_memmap = False

    def __init__(
        self,
        path: str,
        *,
        executor: concurrent.futures.Executor,
        index: T.Sequence[int] | None = None,
        metadata: dict[str, T.Any] | None = None,
    ):
        self._path = path
        self._indices = _find_memmap_indices(path) if index is None else index
        self._executor = executor
        # NOTE: metadata is the same for all tensordicts in the directory

        if metadata is None:
            if len(self._indices) > 0:
                self._metadata = _load_tensordict_metadata(
                    file_io.Path(path) / str(self._indices[0]) / "meta.json"
                )
            else:
                self._metadata = {}
        else:
            self._metadata = metadata

    @property
    @TX.override
    def batch_size(self) -> torch.Size:
        return torch.Size([len(self._indices)])

    @TX.override
    def _index_tensordict(self, index, *args, **kwargs) -> T.Self:
        return self.__class__(
            self._path, executor=self._executor, index=index, metadata=self._metadata
        )

    # ---------------------- #
    # Reading mapped tensors #
    # ---------------------- #

    def _get_meta(
        self, key: str, required: bool = False
    ) -> MemoryMappedTensorMeta | None:
        meta = T.cast(MemoryMappedTensorMeta | None, self._metadata.get(key))
        if meta is None and required:
            msg = f"Key {key} not found in {self}"
            raise KeyError(msg)
        return meta

    @TX.override
    def _get_str(self, key: str, default: Tensor = _NO_DEFAULT) -> Tensor:
        meta = self._get_meta(key, default is _NO_DEFAULT)
        if meta is None:
            return default
        view = LazyStackedMemmapTensorView(
            self._path,
            key,
            self._indices,
            dtype=meta["dtype"],
            shape=meta["shape"],
            executor=self._executor,
        )
        return view.to_tensor()

    @TX.override
    def _get_at_str(
        self, key: str, idx: int | _IndexType, default: Tensor = _NO_DEFAULT
    ) -> Tensor:
        meta = self._get_meta(key, default is _NO_DEFAULT)
        if meta is None:
            return default
        view = LazyStackedMemmapTensorView(
            self._path,
            key,
            self._indices,
            meta["dtype"],
            meta["shape"],
            executor=self._executor,
        )
        return view[idx]

    @TX.override
    def _get_tuple(self, key, default) -> Tensor:
        key_str, *other = key
        if len(other) > 0:
            msg = f"Nested keys are not supported, got: {key}"
            raise NotImplementedError(msg)
        return self._get_str(key_str, default)

    @TX.override
    def _get_at_tuple(self, key, idx, default) -> Tensor:
        key_str, *other = key
        if len(other) > 0:
            msg = f"Nested keys are not supported, got: {key}"
            raise NotImplementedError(msg)
        return self._get_at_str(key_str, idx, default)

    # ----------------------- #
    # Functorch compatablilty #
    # ----------------------- #

    @TX.override
    def _add_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError("Cannot add batch dim to LazyConcatenatedMemmapTensorDict")

    @TX.override
    def _remove_batch_dim(self, *, in_dim, vmap_level):
        raise RuntimeError(
            "Cannot remove batch dim from LazyConcatenatedMemmapTensorDict"
        )

    # ------------ #
    # Dict methods #
    # ------------ #

    @TX.override
    def __setitem__(self, key: str, value: T.Any) -> T.NoReturn:
        raise NotImplementedError()

    @TX.override
    def keys(self) -> T.Iterator[str]:
        yield from (
            k
            for k, v in self._metadata.items()
            if isinstance(v, dict) and "dtype" in v and "shape" in v
        )

    @TX.override
    def values(self) -> T.Iterator[LazyStackedMemmapTensorView]:
        for _, value in self.items():
            yield value

    @TX.override
    def items(self) -> T.Iterator[T.Tuple[str, T.Any]]:
        for key in self.keys():
            yield key, LazyStackedMemmapTensorView(
                self._path,
                key,
                self._indices,
                self._metadata[key]["dtype"],
                self._metadata[key]["shape"],
                executor=self._executor,
            )

    # ---------- #
    # Module API #
    # ---------- #

    @classmethod
    @TX.override
    def _from_module(cls, *args, **kwargs):
        raise NotImplementedError(f"{cls.__name__} does not support from_module")

    @TX.override
    def _to_module(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support to_module"
        )

    # ---------- #
    # State dict #
    # ---------- #

    @TX.override
    def state_dict(
        self, destination=None, prefix="", keep_vars=False, flatten=False
    ) -> collections.OrderedDict[str, T.Any]:
        source = self
        out = collections.OrderedDict()
        out[prefix + "__path"] = source._path
        out[prefix + "__index"] = source._indices
        out[prefix + "__executor"] = source._executor
        if destination is not None:
            destination.update(out)
            return destination
        return out

    @TX.override
    def load_state_dict(
        self,
        state_dict: collections.OrderedDict[str, T.Any],
        strict=True,
        assign=False,
        from_flatten=False,
    ) -> T.Self:
        if assign:
            self._path = state_dict["__path"]
            return self
        else:
            return self.__class__(state_dict["__path"], state_dict["__index"])

    @TX.override
    def share_memory_(self):
        msg = f"{self.__class__.__name__} does not support share_memory_"
        raise NotImplementedError(msg)

    # ---------- #
    # Assignment #
    # ---------- #
    def __error_is_locked(self, *args, **kwargs) -> T.NoReturn:
        msg = f"{self.__class__.__name__} is locked"
        raise RuntimeError(msg)

    _set = __error_is_locked
    _set_str = __error_is_locked
    _set_tuple = __error_is_locked
    _set_at = __error_is_locked
    _set_at_str = __error_is_locked
    _set_at_tuple = __error_is_locked
    __setitem__ = __error_is_locked
    _set_non_tensor = __error_is_locked

    _update = __error_is_locked
    _update_at = __error_is_locked

    # ---------- #
    # Memmap API #
    # ---------- #
    def __error_is_memmaped(self, *args, **kwargs) -> T.NoReturn:
        msg = f"{self.__class__.__name__} is already memory-mapped"
        raise RuntimeError(msg)

    _memmap_ = __error_is_memmaped
    memmap_ = __error_is_memmaped
    memmap = __error_is_memmaped
    memmap_like = __error_is_memmaped
    load_memmap = __error_is_memmaped

    # ----------- #
    # Unsupported #
    # ----------- #

    def __eq__(self, *args, **kwargs):
        return NotImplementedError("Method __eq__ is not supported!")

    def __ne__(self, *args, **kwargs):
        return NotImplementedError("Method __ne__ is not supported!")

    def __or__(self, *args, **kwargs):
        return NotImplementedError("Method __or__ is not supported!")

    def __xor__(self, *args, **kwargs):
        return NotImplementedError("Method __xor__ is not supported!")

    def _apply_nest(self, *args, **kwargs):
        return NotImplementedError("Method _apply_nest is not supported!")

    def _change_batch_size(self, *args, **kwargs):
        return NotImplementedError("Method _change_batch_size is not supported!")

    def _check_device(self, *args, **kwargs):
        return NotImplementedError("Method _check_device is not supported!")

    def _check_is_shared(self, *args, **kwargs):
        return NotImplementedError("Method _check_is_shared is not supported!")

    def _clone(self, *args, **kwargs):
        return NotImplementedError("Method _clone is not supported!")

    def _convert_to_tensordict(self, *args, **kwargs):
        return NotImplementedError("Method _convert_to_tensordict is not supported!")

    def _erase_names(self, *args, **kwargs):
        return NotImplementedError("Method _erase_names is not supported!")

    def _exclude(self, *args, **kwargs):
        return NotImplementedError("Method _exclude is not supported!")

    def _has_names(self, *args, **kwargs):
        return NotImplementedError("Method _has_names is not supported!")

    def _load_memmap(self, *args, **kwargs):
        return NotImplementedError("Method _load_memmap is not supported!")

    def _permute(self, *args, **kwargs):
        return NotImplementedError("Method _permute is not supported!")

    def _rename_subtds(self, *args, **kwargs):
        return NotImplementedError("Method _rename_subtds is not supported!")

    def _select(self, *args, **kwargs):
        return NotImplementedError("Method _select is not supported!")

    def _squeeze(self, *args, **kwargs):
        return NotImplementedError("Method _squeeze is not supported!")

    def _stack_onto_(self, *args, **kwargs):
        return NotImplementedError("Method _stack_onto_ is not supported!")

    def _transpose(self, *args, **kwargs):
        return NotImplementedError("Method _transpose is not supported!")

    def _unbind(self, *args, **kwargs):
        return NotImplementedError("Method _unbind is not supported!")

    def _unsqueeze(self, *args, **kwargs):
        return NotImplementedError("Method _unsqueeze is not supported!")

    def _view(self, *args, **kwargs):
        return NotImplementedError("Method _view is not supported!")

    def all(self, *args, **kwargs):
        return NotImplementedError("Method all is not supported!")

    def any(self, *args, **kwargs):
        return NotImplementedError("Method any is not supported!")

    def contiguous(self, *args, **kwargs):
        return NotImplementedError("Method contiguous is not supported!")

    def del_(self, *args, **kwargs):
        return NotImplementedError("Method del_ is not supported!")

    def detach_(self, *args, **kwargs):
        return NotImplementedError("Method detach_ is not supported!")

    def device(self, *args, **kwargs):
        return NotImplementedError("Method device is not supported!")

    def entry_class(self, *args, **kwargs):
        return NotImplementedError("Method entry_class is not supported!")

    def expand(self, *args, **kwargs):
        return NotImplementedError("Method expand is not supported!")

    def is_contiguous(self, *args, **kwargs):
        return NotImplementedError("Method is_contiguous is not supported!")

    def masked_fill(self, *args, **kwargs):
        return NotImplementedError("Method masked_fill is not supported!")

    def masked_fill_(self, *args, **kwargs):
        return NotImplementedError("Method masked_fill_ is not supported!")

    def masked_select(self, *args, **kwargs):
        return NotImplementedError("Method masked_select is not supported!")

    def names(self, *args, **kwargs):
        return NotImplementedError("Method names is not supported!")

    def pin_memory(self, *args, **kwargs):
        return NotImplementedError("Method pin_memory is not supported!")

    def rename_key_(self, *args, **kwargs):
        return NotImplementedError("Method rename_key_ is not supported!")

    def reshape(self, *args, **kwargs):
        return NotImplementedError("Method reshape is not supported!")

    def split(self, *args, **kwargs):
        return NotImplementedError("Method split is not supported!")

    def to(self, *args, **kwargs):
        return NotImplementedError("Method to is not supported!")


class MemmapTensordictWriter:
    """
    Writes results to a MemmapTensor directory.

    See Also
    --------
    - TensorDict documentation: https://pytorch.org/tensordict/saving.html
    """

    __slots__ = ("is_closed", "path", "_size", "_futures", "_td_cursor", "_executor")

    def __init__(self, path: str, size: int, write_offset: int):
        """
        Parameters
        ----------
        path : str
            The path to the MemmapTensor directory.
        size : int
            The size of the first dimension of the results.
        """
        self.path = file_io.Path(path)
        self.is_closed = False
        self._size: T.Final = size
        self._futures = []
        self._td_cursor = write_offset

    @TX.override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path!r}, size={len(self)}, cursor={self._td_cursor})"

    def __len__(self) -> int:
        return self._size

    def recover(self) -> bool:
        """
        Check whether a previous run has already written all the required results
        to our path.

        If recovered, we automatically close.

        Returns
        -------
        bool
            True when recovered, False when not
        """
        if self.is_closed:
            msg = f"{self.__class__.__name__} is closed"
            raise RuntimeError(msg)

        num_tensordict_subdirs = len(
            [
                d
                for d in self.path.iterdir()
                if d.is_dir() and (d / "meta.json").is_file()
            ]
        )

        barrier()

        if check_main_process():
            if num_tensordict_subdirs == len(self):
                self.close()
                return True
            for p in self.path.iterdir():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)

        barrier()

        return False

    def flush(self):
        """
        Write the results to disk.
        """
        if self.is_closed:
            msg = f"{self.__class__.__name__} is closed"
            raise RuntimeError(msg)
        concurrent.futures.wait(self._futures)
        self.close()

    def add(self, data: TensorDictBase, executor: concurrent.futures.Executor):
        """
        Add an item to the results list, and write to disk if the buffer is full.

        Parameters
        ----------
        data : TensorDictBase
            The data to add.
        """
        if self.is_closed:
            msg = f"{self.__class__.__name__} is closed"
            raise RuntimeError(msg)

        assert (
            data.batch_dims == 1
        ), f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

        for batched_item in data.cpu():
            futures = []
            batched_item._memmap_(
                prefix=self.path / str(self._td_cursor),
                inplace=True,
                like=False,
                futures=futures,
                executor=executor,
                copy_existing=False,
            )

            self._futures.extend(futures)
            self._td_cursor += 1

        self._futures = _wait_until_futures_below_limit(self._futures)

    def close(self):
        if self.is_closed:
            return
        self.is_closed = True

    def read(self, executor: concurrent.futures.Executor) -> TensorDictBase:
        if not self.is_closed:
            msg = "ResultsWriter is not closed"
            raise RuntimeError(msg)
        return LazyStackedMemmapTensorDict(
            str(self.path), executor=executor, index=list(range(len(self)))
        )


@functools.cache
def _get_concurrent_write_limit() -> int:
    """
    Get the maximum number of concurrent writes to the file system.

    Reads the environment variable ``UP_ENGINE_WRITER_CONCURRENT_WRITE_LIMIT``, by
    default set to the amount of CPUs available times 2 on the system.
    This is used as a proxy for the number of concurrent writes that can fit in memory
    without causing a significant performance hit.

    Returns
    -------
    int
        The maximum number of concurrent writes.
        Negative values indicate no limit is imposed.
    """
    from unipercept.config import get_env

    return get_env(
        int, "UP_ENGINE_WRITER_CONCURRENT_WRITE_LIMIT", default=cpus_available() * 2
    )


def _wait_until_futures_below_limit(
    futures: list[concurrent.futures.Future],
) -> list[concurrent.futures.Future]:
    futures = [f for f in futures if not f.done()]
    limit = _get_concurrent_write_limit()
    if limit < 0:
        return futures
    num_exceeded = len(futures) - limit
    if num_exceeded > 0:
        for _ in range(num_exceeded):
            _, not_done = concurrent.futures.wait(
                futures, timeout=None, return_when=concurrent.futures.FIRST_COMPLETED
            )
            futures = list(not_done)
    return futures


def _find_memmap_indices(path: Pathable) -> T.Tuple[int, int]:
    """
    Find the indices of the first and last memory-mapped file in a directory.

    Parameters
    ----------
    path : str
        The path to the directory.

    Returns
    -------
    T.Tuple[int, int]
        The first and last indices.
    """
    path = file_io.Path(path)
    indices = sorted(
        int(p.name)
        for p in path.iterdir()
        if int(p.name) > 0 and (p / "meta.json").is_file()
    )
    return indices[0], indices[-1]


def _load_tensordict_metadata(path: Pathable):
    with open(file_io.Path(path)) as json_metadata:
        metadata = json.load(json_metadata)
    return metadata


def _estimate_memory_footprint(o, handlers={}, verbose=False):
    """
    Calculates the approximate memory footprint an object and all of its contents.
    """
    dict_handler = lambda d: I.chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        collections.deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
        TensorDictBase: dict_handler,
        Tensorclass: dict_handler,
        T.Sequence: iter,
        T.Mapping: dict_handler,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = sys.getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=sys.stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)
