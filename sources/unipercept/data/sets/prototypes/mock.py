from __future__ import annotations

from torchdata.datapipes.iter import IterDataPipe
from typing_extensions import override
from unicore import catalog, datapipes, file_io


@catalog.register
class MockDataset(IterDataPipe):
    def __init__(self, *, length: int = 10):
        self.length = length

    @override
    def __iter__(self):
        yield from range(self.length)
