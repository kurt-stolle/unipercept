"""
Implements a handler for writing results to a file from multiple processes.
"""

from __future__ import annotations

import functools
import typing as T

import accelerate
import torch
import torch.types
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


class ResultsWriter(T.Protocol):
    def __init__(self, path: str, size: int):
        ...

    def add(self, data: TensorDictBase):
        ...

    @property
    def path(self) -> file_io.Path:
        ...

    @property
    def tensordict(self) -> TensorDictBase:
        ...

    def __len__(self) -> int:
        ...


class PersistentTensordictWriter:
    """
    Writes results to a H5 file using PersistentTensorDict from multiple processes, uses a buffer to reduce the number of writes.
    """

    __slots__ = ["_td_cursor", "_td", "_path", "_td_factory", "_buffer_list", "_buffer_size"]

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
        self._path: T.Final = path
        self._buffer_list: list[TensorDictBase] = []
        self._buffer_size = buffer_size
        self._td: PersistentTensorDict | None = None
        self._td_cursor = 0
        self._td_factory = functools.partial(
            PersistentTensorDict, batch_size=[size], mode="w" if check_main_process() else "r"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)

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

    @on_main_process()
    def _append(self, data: TensorDictBase) -> None:
        self._buffer_list.append(data.cpu())
        if len(self._buffer_list) == self._buffer_size:
            self.write()

    @on_main_process()
    def write(self):
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

    @property
    def path(self) -> file_io.Path:
        return file_io.Path(self._path)

    @property
    def tensordict(self) -> TensorDictBase:
        if self._td is None:
            self._td = self._td_factory(filename=self.path)
        return self._td

    def __del__(self) -> None:
        self.close()

    def close(self):
        if self._td is not None:
            _logger.debug("Closing results writer")

            self._td.close()
            self._td = None
