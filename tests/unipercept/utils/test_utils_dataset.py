from __future__ import annotations

import typing as T

import pytest
import typing_extensions as TX
from typing_extensions import override

from unipercept.utils.dataset import Dataset
from unipercept.utils.frozendict import frozendict


class MyDataset(Dataset, info=lambda: {"name": "FooBar"}):
    @override
    def _load_data(self, id, item, info):
        if item["id"] == 1:
            return item | {"name": "Foo"}
        elif item["id"] == 2:
            return item | {"name": "Bar"}
        else:
            return item | {"name": "Baz"}

    @override
    def _build_manifest(self):
        return {"items": [{"id": 1}, {"id": 2}]}

    @staticmethod
    def gatherer(manifest):
        for i, v in enumerate(manifest["items"]):
            yield str(i), v


@pytest.fixture
def dataset():
    ds = MyDataset(queue_fn=MyDataset.gatherer)
    return ds


def test_dataset(dataset):
    from typing import Iterable, Mapping

    # Manifest
    assert isinstance(dataset.manifest, Iterable)

    mfst = dict(dataset.manifest)

    assert len(mfst["items"]) == 2
    assert isinstance(mfst, Mapping)
    assert isinstance(mfst["items"], Iterable)
    assert mfst["items"] == [{"id": 1}, {"id": 2}]

    # Queue
    assert isinstance(dataset.queue, Iterable)

    # Datapipe
    assert isinstance(dataset.datapipe, Iterable)
    assert list(dataset.datapipe) == [
        {"id": 1, "name": "Foo"},
        {"id": 2, "name": "Bar"},
    ]
