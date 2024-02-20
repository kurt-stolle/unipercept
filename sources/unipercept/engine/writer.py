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
import torch.multiprocessing as M
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
from unipercept.utils.typings import Pathable
from unipercept.state import (
    barrier,
    check_main_process,
    gather_tensordict,
    on_main_process,
    main_process_first,
    cpus_available,
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
    def flush(self) -> None:
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
        self._pool = concurrent.futures.ThreadPoolExecutor()
        self._futures = []

        self._td: TensorDict | None = None
        self._td_cursor = 0

    @functools.cached_property
    def path(self) -> file_io.Path:
        p = file_io.Path(self._path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def __len__(self) -> int:
        return self._size

    @on_main_process()
    def _append(self, data: TensorDictBase) -> None:
        for batched_item in data.cpu():
            batched_item._memmap_(
                prefix=self.path / str(self._td_cursor),
                inplace=True,
                like=False,
                futures=self._futures,
                executor=self._pool,
                copy_existing=False,
            )

            self._td_cursor += 1

    @on_main_process()
    def _write(self):
        _logger.debug("Waiting for all tasks to finish writing")
        concurrent.futures.wait(self._futures)

    @TX.override
    def flush(self):
        """
        Write the results to disk.
        """
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")

        with main_process_first():
            self._write()
        self.close()
        barrier()

    @TX.override
    def add(self, data: TensorDictBase):
        """
        Add an item to the results list, and write to disk if the buffer is full.

        Parameters
        ----------
        data : TensorDictBase
            The data to add.
        """
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")

        assert (
            data.batch_dims == 1
        ), f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

        data = gather_tensordict(data)
        self._append(data)

    @TX.override
    def close(self):
        self._is_closed = True
        self._pool.shutdown(wait=True)

    @property
    @TX.override
    def tensordict(self) -> TensorDictBase:
        if not self._is_closed:
            raise RuntimeError("ResultsWriter is not closed")
        if self._td is None:
            self._td = LazyStackedTensorDict._load_memmap(self.path, {"stack_dim": 0})
        return self._td


class PersistentTensordictWriterProcess(M.Process):
    def __init__(
        self,
        path: Pathable,
        size: int,
        compression: T.Literal["lzf", "gzip"] | None,
        compression_opts: T.Any,
        queue: M.Queue,
    ):
        super().__init__()
        self.path = file_io.Path(path)
        self.size = size
        self.compression = compression
        self.compression_opts = compression_opts
        self.queue = queue
        self.cursor = 0

    @TX.override
    def run(self):
        _logger.debug("Starting H5 results writer process!")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = PersistentTensorDict(
            filename=self.path,
            batch_size=[self.size],
            mode="w",
            compression=self.compression,
            compression_opts=self.compression_opts,
        )

        try:
            while True:
                data = self.queue.get()
                if data is None:
                    _logger.debug("Received None, closing writer")
                    break
                writer[self.cursor : self.cursor + len(data)] = data
                self.cursor += len(data)
        finally:
            writer.close()


class PersistentTensordictWriter(ResultsWriter):
    """
    Writes results to a H5 file using PersistentTensorDict from multiple processes, uses a buffer to reduce the number of writes.
    """

    def __init__(
        self,
        path: Pathable,
        size: int,
        buffer_size: int = -1,
        compression: T.Literal["lzf", "gzip"] | None ="lzf",
        compression_opts: T.Any = None,
    ):
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
        self._path = file_io.Path(path)
        if self._path.is_dir():
            self._path = self._path / "results.h5"
        elif self._path.suffix not in (".h5", ".hdf5"):
            self._path = self._path.with_suffix(".h5")

        if check_main_process():
            self._queue = M.Queue(buffer_size if buffer_size > 0 else cpus_available() * 2)
            self._writer = PersistentTensordictWriterProcess(
                path, size, compression, compression_opts, queue=self._queue
            )
            self._writer.start()
        else:
            self._queue = None
            self._writer = None

        self._td: PersistentTensorDict | None = None
        self._td_factory = functools.partial(
            PersistentTensorDict,
            batch_size=[len(self)],
            mode="r",
            compression=compression,
            compression_opts=compression_opts,
        )
    def __del__(self):
        if not self._is_closed:
            _logger.warning("ResultsWriter was not closed, closing now", stacklevel=2)
        self.close()

    def __len__(self) -> int:
        return self._size

    @on_main_process()
    def _append(self, data: TensorDictBase) -> None:
        if self._queue is None:
            raise RuntimeError("ResultsWriter queue is None")
        self._queue.put(data.cpu())

    @TX.override
    def flush(self):
        """
        Write the results to disk.
        """
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")
        
        if check_main_process():
            assert self._queue is not None and self._writer is not None
            _logger.debug("Sending close signal to H5 writer")
            self._queue.put(None)
            _logger.debug("Waiting for H5 writer to finish")
            self._writer.join()
            self._writer.close()
            self._writer = None
        barrier()

        self.close()

    @TX.override
    def add(self, data: TensorDictBase):
        """
        Add an item to the results list, and write to disk if the buffer is full.

        Parameters
        ----------
        data : TensorDictBase
            The data to add.
        """
        if self._is_closed:
            raise RuntimeError(f"{self.__class__.__name__} is closed")

        assert (
            data.batch_dims == 1
        ), f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

        data = gather_tensordict(data)
        self._append(data)

    @TX.override
    def close(self):
        self._is_closed = True
        
        if check_main_process():
            if self._queue is not None:
                self._queue.close()
                self._queue = None
            if self._writer is not None:
                self._writer.terminate()
                self._writer.close()
                self._writer = None

        if self._td is not None:
            self._td.close()
            self._td = None

    @property
    @TX.override
    def tensordict(self) -> TensorDictBase:
        if not self._is_closed:
            raise RuntimeError("ResultsWriter is not closed")
        if self._td is None:
            self._td = self._td_factory(filename=self._path)
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
