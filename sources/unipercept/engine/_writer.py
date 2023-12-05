"""
Implements a handler for writing results to a file from multiple processes.
"""

from __future__ import annotations

from tensordict import PersistentTensorDict, TensorDictBase

from unipercept.state import main_process_first, check_main_process, on_main_process

__all__ = ["ResultsWriter"]


class ResultsWriter:
    @main_process_first()
    def __init__(self, filename: str, size: int):
        self._td_cursor = 0
        self._td = PersistentTensorDict(filename=filename, batch_size=[size], mode="w" if check_main_process() else "r")

    @on_main_process()
    def add(self, data: TensorDictBase):
        off_l = self._td_cursor
        off_h = off_l + data.batch_size[0]

        self._td[off_l:off_h] = data
        self._td_cursor = off_h

    @property
    def tensordict(self) -> TensorDictBase:
        return self._td

    @property
    def cursor(self) -> int:
        return self._td_cursor
