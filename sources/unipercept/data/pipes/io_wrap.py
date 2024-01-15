from __future__ import annotations

import os
from typing import Any, Callable, Iterator, Sequence

from torch.utils.data.datapipes.utils.common import match_masks
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IoPathFileLister, IoPathFileOpener, IoPathSaver, IterDataPipe
from typing_extensions import override

from unicore import file_io

__all__ = ["UniCoreFileLister", "UniCoreFileOpener", "UniCoreSaver"]


@functional_datapipe("list_files_by_unicore")
class UniCoreFileLister(IoPathFileLister):
    """
    See ``IoPathFileLister``.
    """

    def __init__(self, root: str | Sequence[str] | IterDataPipe, masks: str | list[str], recursive=False) -> None:
        super().__init__(root, masks, pathmgr=file_io._manager)

        self.recursive = recursive

    @override
    def __iter__(self) -> Iterator[str]:
        for path in self.datapipe:
            yield from self._walk(path)

    def _walk(self, path: str) -> Iterator[str]:
        if self.pathmgr.isfile(path):
            yield path
            return
        for file_name in self.pathmgr.ls(path):
            file_path = os.path.join(path, file_name)
            if self.pathmgr.isdir(file_path):
                if self.recursive:
                    yield from self._walk(file_path)
                else:
                    continue
            elif match_masks(file_name, self.masks):
                yield os.path.join(path, file_name)


@functional_datapipe("open_files_by_unicore")
class UniCoreFileOpener(IoPathFileOpener):
    """
    See ``IoPathFileOpener``.
    """

    def __init__(self, source_datapipe: IterDataPipe[str], mode: str = "r") -> None:
        super().__init__(source_datapipe, mode, pathmgr=file_io._manager)


@functional_datapipe("save_by_unicore")
class UniCoreSaver(IoPathSaver):
    """
    See ``UniCoreSaver``.
    """

    def __init__(
        self,
        source_datapipe: IterDataPipe[tuple[Any, bytes | bytearray | str]],
        mode: str = "w",
        filepath_fn: Callable | None = None,
    ):
        super().__init__(source_datapipe, mode, filepath_fn, pathmgr=file_io._manager)
