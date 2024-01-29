from __future__ import annotations

import typing as T

import pytest
import typing_extensions as TX

from unipercept import get_dataset, get_info


@pytest.mark.parametrize("split", ["train", "val", "test", "trainval"])
@pytest.mark.parametrize("year", ["2012", "2007"])
def test_voc_dataset(split, year):
    ds = get_dataset("pascal_voc")(split=split, year=year, download=False)

    assert ds.split == split
    assert ds.year == year
