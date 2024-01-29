from __future__ import annotations

import typing as T

import typing_extensions as TX

from unipercept import file_io


def test_file_io_globals():
    for d in dir(file_io):
        assert getattr(file_io, d) is not None
