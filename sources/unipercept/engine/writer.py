"""
Implements a handler for writing results to a file from multiple processes.
"""

from __future__ import annotations

import typing as T

import torch
from tensordict import PersistentTensorDict, TensorDictBase
from unicore import file_io

from unipercept.log import get_logger
from unipercept.state import check_main_process, main_process_first, on_main_process

__all__ = ["ResultsWriter", "PersistentTensordictWriter"]

_logger = get_logger(__name__)


class ResultsWriter(T.Protocol):
    def __init__(self, path: str, size: int):
        ...

    @on_main_process()
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
    __slots__ = ["_cursor", "_td", "_path", "_size", "_buffer"]

    def __init__(self, path: str, size: int, buffer_size: int = 10):
        self._cursor = 0
        self._path: T.Final = path
        self._size: T.Final = size
        self._buffer: T.List[TensorDictBase] = []
        self._td = None

        self.path.parent.mkdir(parents=True, exist_ok=True)

    @on_main_process()
    def add(self, data: TensorDictBase):
        assert data.batch_dims == 1, "ResultsWriter only supports 1D batches."
        self._buffer.append(data)

    @on_main_process()
    def flush(self):
        if len(self._buffer) > 0:
            _logger.debug("Writing results to storage")
            data = torch.cat(self._buffer, dim=0)  # type: ignore
            self._write_to_disk(data)
            self._buffer.clear()
        else:
            _logger.debug("No results to write")

    def _write_to_disk(self, data: TensorDictBase):
        off_l = self._cursor
        off_h = off_l + data.batch_size[0]

        self.tensordict[off_l:off_h] = data
        self._cursor = off_h

    @property
    def path(self) -> file_io.Path:
        return file_io.Path(self._path)

    @property
    def tensordict(self) -> TensorDictBase:
        if self._td is None:
            self._td = PersistentTensorDict(
                filename=self.path, batch_size=[self._size], mode="w" if check_main_process() else "r"
            )
        return self._td

    def __len__(self) -> int:
        return self._size

    def __del__(self):
        self.close()

    def close(self):
        if self._td is not None:
            self._td.close()
            self._td = None
