from __future__ import annotations

import re
from dataclasses import dataclass

from unipercept.utils.matchable import Matchable


def test_matchable_class():
    pattern = re.compile(r"^(?P<value_one>\w+)-(?P<value_two>\w+)$")

    @dataclass
    class Foo(Matchable, match_groups=["value_one", "value_two"]):
        value_one: str
        value_two: str

    match = pattern.match("foo-bar")
    assert match is not None
    foo = Foo.from_match(match)
    assert foo.value_one == "foo"
    assert foo.value_two == "bar"

    @dataclass
    class Bar(
        Matchable,
        match_groups=["value_one"],
        match_kwgroups={"value_three": "value_two"},
    ):
        value_one: str
        value_three: str

    match = pattern.match("foo-bar")
    assert match is not None
    bar = Bar.from_match(match)
    assert bar.value_one == "foo"
    assert bar.value_three == "bar"
