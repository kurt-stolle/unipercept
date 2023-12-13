"""
Implements a handler for writing results to a file from multiple processes.
"""

from __future__ import annotations

import functools
import typing as T
import typing_extensions as TX

import accelerate
import torch
import torch.types
import abc
import tempfile
import functools
from tensordict import PersistentTensorDict, TensorDict, TensorDictBase
from unicore import file_io

from unipercept.log import get_logger
from unipercept.state import (
    barrier,
    check_main_process,
    gather_tensordict,
    get_process_count,
    get_process_index,
    main_process_first,
    on_main_process,
)

__all__ = ["ResultsWriter", "PersistentTensordictWriter"]

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
            PersistentTensorDict, batch_size=[len(self)], mode="w" if check_main_process() else "r"
        )

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
        if len(self._buffer_list) == self._buffer_size:
            self.write()

    @on_main_process()
    def _write(self):
        data = torch.cat(self._buffer_list, dim=0)  # type: ignore

        assert data.batch_dims == 1, f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

        # Determine offsets in target storage
        off_l = self._td_cursor
        off_h = off_l + data.batch_size[0]

        if off_l == off_h:
            _logger.debug("Nothing to write, skipping")
            return

        _logger.debug(f"Writing {data.batch_size} results to storage")
        # Perform write operation
        self.tensordict[off_l:off_h] = data

        # Update cursor to the next available offset
        self._td_cursor = off_h
        self._buffer_list.clear()

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

        assert data.batch_dims == 1, f"ResultsWriter only supports 1D batches. Got {data.batch_dims}."

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
