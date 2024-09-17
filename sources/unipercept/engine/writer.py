"""
Implements a handler for writing results to a file from multiple processes.
"""

from __future__ import annotations

import collections
import concurrent.futures
import functools
import itertools as I
import json
import shutil
import sys
import typing as T

import torch
import torch.types
import typing_extensions as TX
from tensordict import TensorDictBase
from torch import Tensor

from unipercept import file_io
from unipercept.log import get_logger
from unipercept.state import cpus_available, main_process_first
from unipercept.types import Pathable
from unipercept.utils.tensorclass import Tensorclass

_logger = get_logger(__name__)

_NO_DEFAULT: T.Final = T.cast(torch.Tensor, "_no_default_")
_IndexType: T.TypeAlias = torch.Tensor | slice | T.Iterable[int]


class MemoryMappedTensorMeta(T.TypedDict):
    """
    Metadata for tensors saved with every tensordict in a memory-mapped directory.
    """

    dtype: torch.dtype
    shape: torch.Size


class ResultsReaderTensorView:
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
        if isinstance(view, T.Iterable):
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
        return torch.from_file(
            str(path),
            shared=False,
            dtype=self.dtype,
            size=self.shape.numel(),
            requires_grad=False,
        ).reshape(self.shape)

        # return tensor.clone(memory_format=torch.contiguous_format)

    def to_tensor(self) -> Tensor:
        return self[:]


class ResultsReader:
    """
    This tensordict is a read-only view of a dictory with containig previously
    memory-mapped tensordicts.
    """

    __slots__ = ("_path", "_indices", "_metadata", "_executor")

    def __init__(
        self,
        path: str,
        *,
        executor: concurrent.futures.Executor | None = None,
        index: T.Sequence[int] | None = None,
        metadata: dict[str, T.Any] | None = None,
    ):
        self._path = path
        self._indices = index or self._read_indices()
        self._executor = executor or concurrent.futures.ThreadPoolExecutor()

        # NOTE: metadata is the same for all tensordicts in the directory
        if metadata is None:
            self._metadata = self._read_metadata()
        else:
            self._metadata = metadata

    @property
    def path(self) -> str:
        return self._path

    @path.setter
    def path(self, path: str):
        self._path = path
        self._indices = self._read_indices()
        self._metadata = self._read_metadata()

    def _read_indices(self):
        return _find_memmap_indices(self._path)

    def _read_metadata(self):
        if len(self) == 0:
            return {}
        return _load_tensordict_metadata(
            file_io.Path(self._path) / str(self._indices[0]) / "meta.json"
        )

    @T.override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self._path!r})[:{len(self)}]"

    @T.override
    def __len__(self) -> int:
        return len(self._indices)

    def get_meta(
        self, key: str, required: bool = False
    ) -> MemoryMappedTensorMeta | None:
        meta = T.cast(MemoryMappedTensorMeta | None, self._metadata.get(key))
        if meta is None and required:
            msg = f"Key {key} not found in {self}"
            raise KeyError(msg)
        return meta

    def get(self, key: str, default: Tensor = _NO_DEFAULT) -> Tensor:
        meta = self.get_meta(key, default is _NO_DEFAULT)
        if meta is None:
            return default
        view = ResultsReaderTensorView(
            self._path,
            key,
            self._indices,
            dtype=meta["dtype"],
            shape=meta["shape"],
            executor=self._executor,
        )
        return view.to_tensor()

    def get_at(
        self, key: str, idx: int | _IndexType, default: Tensor = _NO_DEFAULT
    ) -> Tensor:
        meta = self.get_meta(key, default is _NO_DEFAULT)
        if meta is None:
            return default
        view = ResultsReaderTensorView(
            self._path,
            key,
            self._indices,
            meta["dtype"],
            meta["shape"],
            executor=self._executor,
        )
        return view[idx]

    def keys(self) -> T.Iterator[str]:
        yield from (
            k
            for k, v in self._metadata.items()
            if isinstance(v, dict) and "dtype" in v and "shape" in v
        )

    def values(self) -> T.Iterator[ResultsReaderTensorView]:
        for _, value in self.items():
            yield value

    def items(self) -> T.Iterator[tuple[str, T.Any]]:
        for key in self.keys():
            yield (
                key,
                ResultsReaderTensorView(
                    self._path,
                    key,
                    self._indices,
                    self._metadata[key]["dtype"],
                    self._metadata[key]["shape"],
                    executor=self._executor,
                ),
            )


class ResultsWriter:
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

        with main_process_first():
            if num_tensordict_subdirs == len(self):
                self.close()
                return True
            for p in self.path.iterdir():
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)

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
                share_non_tensor=False,
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
        return ResultsReader(
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
    from unipercept.config.env import get_env

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


def _find_memmap_indices(path: Pathable) -> tuple[int, int]:
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
