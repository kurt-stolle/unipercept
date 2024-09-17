from __future__ import annotations

from typing import reveal_type

import pytest
from unipercept.utils.missing import MissingValue


def test_missing_value_type():
    NA = MissingValue("NA")
    assert NA.__name__ == "NA"
    assert str(NA) == "?NA"

    # Test equality
    OTHER = MissingValue("OTHER")
    assert NA != OTHER
    assert NA == NA

    # Test representation
    assert repr(NA) == "?NA"

    # Test value retrieval/creation
    NEW_NA = MissingValue("NA")
    assert NA is NEW_NA  # They should be the same instance

    # Test value retrieval/creation with different case
    lower_case_na = MissingValue("na")
    assert (
        NA is lower_case_na
    )  # They should be the same instance because of the uppercase conversion

    # Test non-equality with a different type
    assert NA != "NA"


def test_missing_value_gaurd():
    NA = MissingValue("NA")

    def test_type(val: str | type[NA]):
        if NA.is_value(val):
            reveal_type(val)
            assert val is not NA
        elif NA.is_missing(val):
            reveal_type(val)
            assert val is NA
        else:
            pytest.fail("`val` is revealed as both not a value and not missing!")

    test_type(NA)
    test_type("foo")
