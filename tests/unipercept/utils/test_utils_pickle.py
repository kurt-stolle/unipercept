"""
Implements tests for `unipercept.utils.pickleable`.
"""

from __future__ import annotations

import pickle
from functools import partial

import pytest
from unipercept.utils.pickle import as_picklable, pickles


def some_method():
    return 1005


def partial_method(v):
    return v


class CallableFunc:
    def __init__(self, v):
        self.v = v

    def __call__(self):
        return self.v


MOCK_FNS = [
    (some_method, 1005, True),
    ((lambda: 1004), 1004, False),
    (partial(partial_method, 1003), 1003, True),
    (CallableFunc(1002), 1002, True),
]


def check_pickled_return(fn, r, does_pickle):
    if does_pickle:
        fn_pkl = pickle.dumps(fn)
        fn_unpkl = pickle.loads(fn_pkl)

        assert fn_unpkl() == r
    else:
        with pytest.raises(pickle.PicklingError):
            pickle.dumps(fn)


@pytest.mark.parametrize("fn, r, should_pickle", MOCK_FNS)
def test_pickles(fn, r, should_pickle):
    """
    Test if an object can be pickled.
    """

    does_pickle = pickles(fn)
    assert does_pickle == should_pickle
    check_pickled_return(fn, r, does_pickle)


@pytest.mark.parametrize(
    "fn, r, should_pickle", filter(lambda tpl: not tpl[2], MOCK_FNS)
)
def test_as_picklable(fn, r, should_pickle):
    """
    Test if an object can be pickled.
    """

    if should_pickle:
        raise RuntimeError("This test is only for non-picklable objects")
    fn_wrap = as_picklable(fn)
    check_pickled_return(fn_wrap, r, True)

    res_original = fn()
    res_wrapped = fn_wrap()

    assert res_original == res_wrapped, f"Expected {res_original}, got {res_wrapped}"
