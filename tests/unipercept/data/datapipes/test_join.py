from __future__ import annotations

import typing as T

import pytest
import typing_extensions as TX
from torchdata.datapipes.iter import IterableWrapper

from unipercept.data.pipes.join import LeftIndexJoin


def test_left_join_pipe():
    ids = IterableWrapper([(1, "a"), (2, "b"), (3, "d")])
    join = LeftIndexJoin(
        ids,
        (IterableWrapper([(1, "a")]), IterableWrapper([(1, "b"), (2, "x")])),  # type: ignore
        target_index=lambda x: x[0],
        source_index=lambda x: x[0],
    )

    it = (m for m in join.__iter__())

    assert next(it) == ((1, "a"), (1, "a"), (1, "b"))
    assert next(it) == ((2, "b"), None, (2, "x"))
    assert next(it) == ((3, "d"), None, None)

    with pytest.raises(StopIteration):
        next(it)
