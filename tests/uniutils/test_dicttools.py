from typing import Sequence, TypedDict

import pytest
from uniutils.dicttools import is_dict_type


class DictAB(TypedDict):
    a: int
    b: str


class DictXY(TypedDict):
    x: int
    y: str


@pytest.mark.unit
def test_is_dict_type_invalid():
    d = {"a": 1, "b": True}
    assert not is_dict_type(d, DictAB)

    d = {"a": 1, "b": 1}
    assert not is_dict_type(d, DictAB)


@pytest.mark.unit
def test_is_dict_type_complete():
    d = {"a": 1, "b": "2"}  # good
    assert is_dict_type(d, DictAB)
    assert not is_dict_type(d, DictXY)

    d["x"] = 3  # extra key
    assert not is_dict_type(d, DictAB)


class DictC(TypedDict, total=False):
    c: int


class DictABC(DictAB, total=False):
    c: int


@pytest.mark.unit
def test_is_dict_type_partial():
    d = {"a": 1, "b": "2", "c": 3}
    assert is_dict_type(d, DictABC)
    assert not is_dict_type(d, DictAB)  # extra key {c}
    assert not is_dict_type(d, DictC)  # extra key {a,b}

    d.pop("c")
    assert is_dict_type(d, DictAB)
    assert is_dict_type(d, DictABC)  # total=False, so this is ok


class DictNested(TypedDict):
    abc: DictABC
    d: float


@pytest.mark.unit
def test_is_dict_type_nested():
    d = {"abc": {"a": 1, "b": "2"}, "d": 3.0}

    assert is_dict_type(d, DictNested)


class DictAbstract(TypedDict):
    z: Sequence[int]


@pytest.mark.unit
def test_is_dict_type_sequence():
    d = {}
    d["z"] = [1, 2, 3]  # ok: list of int
    assert is_dict_type(d, DictAbstract)

    d["z"] = [1, 2, "3"]  # ok: list of int and str, but nested type is not checked
    assert is_dict_type(d, DictAbstract)

    d["z"] = (1, 2, 3)  # ok: tuple of int
    assert is_dict_type(d, DictAbstract)
