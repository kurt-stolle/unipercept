"""
Implements a handler for writing results to a file from multiple processes.
"""

from __future__ import annotations

import abc
import collections
import functools
import gc
import itertools as I
import sys
import typing as T

import torch
import torch.types
import typing_extensions as TX
import concurrent.futures
from tensordict import (
    LazyStackedTensorDict,
    PersistentTensorDict,
    TensorDictBase,
    TensorDict,
    MemoryMappedTensor,
)
from tensordict.utils import TensorDictFuture

from unipercept import file_io
from unipercept.log import get_logger
from unipercept.state import (
    check_main_process,
    gather_tensordict,
    on_main_process,
    main_process_first,
    cpus_available
)
from unipercept.utils.tensorclass import Tensorclass

__all__ = ["ResultsWriter", "PersistentTensordictWriter", "MemmapTensordictWriter"]

_logger = get_logger(__name__)

_P = T.ParamSpec("_P")
_R = T.TypeVar("_R")


class ResultsWriter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def add(self, data: TensorDictBase) -> None:
        raise NotImplementedError("Abstract method `add` not implemented.")

    @abc.abstractmethod
    def write(self) -> None:
        raise NotImplementedError("Abstract method `write` not implemented.")

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError("Abstract method `close` not implemented.")

    @property
    @abc.abstractmethod
    def tensordict(self) -> TensorDictBase:
        raise NotImplementedError("Abstract property `tensordict` not implemented.")


class MemmapTensordictWriter(ResultsWriter):
    """
    Writes results to a MemmapTensor directory.

    See Also
    --------
    - TensorDict documentation: https://pytorch.org/tensordict/saving.html
    """

    def __init__(self, path: str, size: int):
        """
        Parameters
        ----------
        path : str
            The path to the MemmapTensor directory.
        size : int
            The size of the first dimension of the results.
        """
        self._size: T.Final = size
        self._is_closed = False
        self._path = path
        self._pool = concurrent.futures.ProcessPoolExecutor(max_workers=cpus_available())

        self._td: TensorDict | None = None
        self._td_cursor = 0

    @staticmethod
    def error_when_closed(
        fn: T.Callable[T.Concatenate[MemmapTensordictWriter, _P], _R]
    ) -> T.Callable[T.Concatenate[MemmapTensordictWriter, _P], _R]:
        @functools.wraps(fn)
        def wrapper(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            if self._is_closed:
                raise RuntimeError(f"{self.__class__.__name__} is closed")
            return fn(self, *args, **kwargs)

        return wrapper

    @functools.cached_property
    def path(self) -> file_io.Path:
        p = file_io.Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __len__(self) -> int:
        return self._size

    @on_main_process()
    def _append(self, data: TensorDictBase) -> None:
        for batched_item in data:
            batched_item.cpu()._memmap_(
                prefix=self.path / str(self._td_cursor),
                inplace=False,
                like=False,
                futures=[],
                executor=self._pool,
                copy_existing=False,
            )

            self._td_cursor += 1

    @on_main_process()
    def _write(self):
        _logger.debug("Waiting for buffer to finish writing (if necessary)")
        self._pool.shutdown(wait=True)

    @TX.override
    @error_when_closed
    def write(self):
        """
        Write the results to disk.
        """
        with main_process_first():
            self._write()

    @TX.override
    @error_when_closed
    def add(self, data: TensorDictBase):
        """
        Add an item to the results list, and write to disk if the buffer is full.

        Parameters
        ----------
        data : TensorDictBase
            The data to add.
        """

        assert (
            data.batch_dims == 1
        ), f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

        data = gather_tensordict(data)
        self._append(data)

    @TX.override
    @error_when_closed
    def close(self):
        self._is_closed = True
        if self._td is not None:
            self._td = None

    @property
    @TX.override
    @error_when_closed
    def tensordict(self) -> TensorDictBase:
        if self._td is None:
            self._td = LazyStackedTensorDict._load_memmap(self.path, {"stack_dim": 0})
        return self._td


class PersistentTensordictWriter(ResultsWriter):
    """
    Writes results to a H5 file using PersistentTensorDict from multiple processes, uses a buffer to reduce the number of writes.
    """

    def __init__(self, path: str, size: int, buffer_size: int = -1):
        """
        Parameters
        ----------
        path : str
            The path to the H5 file.
        size : int
            The size of the first dimension of the results.
        buffer_size : int, optional
            The size of the buffer, by default -1, which means no buffering.
        """
        self._size: T.Final = size
        self._is_closed = False
        self._path = path

        self._buffer_list: list[TensorDictBase] = []
        self._buffer_size = buffer_size

        self._td: PersistentTensorDict | None = None
        self._td_cursor = 0
        self._td_factory = functools.partial(
            PersistentTensorDict,
            batch_size=[len(self)],
            mode="w" if check_main_process() else "r",
            compression=None,
            compression_opts=None,
        )  # See kwargs: https://docs.h5py.org/en/stable/high/group.html#h5py.Group.create_dataset

    def __del__(self):
        if not self._is_closed:
            _logger.warning("ResultsWriter was not closed, closing now", stacklevel=2)
        self.close()

    @staticmethod
    def error_when_closed(
        fn: T.Callable[T.Concatenate[PersistentTensordictWriter, _P], _R]
    ) -> T.Callable[T.Concatenate[PersistentTensordictWriter, _P], _R]:
        @functools.wraps(fn)
        def wrapper(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
            if self._is_closed:
                raise RuntimeError(f"{self.__class__.__name__} is closed")
            return fn(self, *args, **kwargs)

        return wrapper

    @functools.cached_property
    def path(self) -> file_io.Path:
        p = file_io.Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __len__(self) -> int:
        return self._size

    @on_main_process()
    def _append(self, data: TensorDictBase) -> None:
        self._buffer_list.append(data.cpu())
        if len(self._buffer_list) >= self._buffer_size:
            self._write()

    @on_main_process()
    def _write(self):
        num_items = len(self._buffer_list)
        if num_items == 0:
            _logger.debug("Writable results buffer empty, skipping")
            return

        data = torch.cat(self._buffer_list, dim=0)
        if data.batch_dims != 1:
            msg = f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."
            raise ValueError(msg)
        self._buffer_list.clear()
        gc.collect()

        # Determine offsets in target storage
        off_l = self._td_cursor
        off_h = off_l + data.batch_size[0]
        if off_l == off_h:
            _logger.debug("Writable results shard has no data, skipping")
            return

        self.tensordict[off_l:off_h] = data
        self._td_cursor = off_h

    @TX.override
    @error_when_closed
    def write(self):
        """
        Write the results to disk.
        """
        self._write()

    @TX.override
    @error_when_closed
    def add(self, data: TensorDictBase):
        """
        Add an item to the results list, and write to disk if the buffer is full.

        Parameters
        ----------
        data : TensorDictBase
            The data to add.
        """

        assert (
            data.batch_dims == 1
        ), f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

        data = gather_tensordict(data)
        self._append(data)

    @TX.override
    @error_when_closed
    def close(self):
        self._is_closed = True
        self._buffer_list.clear()

        if self._td is not None:
            self._td.close()
            self._td = None

    @property
    @TX.override
    @error_when_closed
    def tensordict(self) -> TensorDictBase:
        if self._td is None:
            self._td = self._td_factory(filename=self.path)
        return self._td


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
