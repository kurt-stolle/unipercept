from dataclasses import dataclass

import pytest

from unipercept.utils.dataserial import (
    SerializeRegistry,
    serializable,
    serializable_base,
)


@pytest.mark.parametrize("kw_only", [True, False])
@pytest.mark.parametrize("slots", [True, False])
def test_dataclass_serialize_json(kw_only, slots):
    @dataclass(frozen=True, kw_only=True, slots=True)
    class Foo:
        a: int
        b: int

    @dataclass(frozen=True, kw_only=True, slots=True)
    class Bar:
        c: int
        d: Foo

    @serializable
    @dataclass(frozen=True, kw_only=True, slots=True)
    class Baz:
        e: int
        f: Bar

    json_in = '{"e": 1, "f": {"c": 2, "d": {"a": 3, "b": 4}}}'
    baz = Baz.from_json(json_in)
    assert baz.e == 1
    assert baz.f.c == 2
    assert baz.f.d.a == 3
    assert baz.f.d.b == 4

    baz_json = baz.to_json()
    # assert baz_json == json_in
    baz_reverse = Baz.from_json(baz_json)
    assert baz_reverse == baz


def test_dataclass_serialize_registry():
    ctx = SerializeRegistry()

    @ctx.as_target
    @dataclass
    class Foo:
        a: int
        b: int

    @ctx.as_target
    @serializable
    @dataclass
    class Bar:
        c: int
        d: int

    @ctx.as_source
    @serializable
    @dataclass
    class Baz:
        pass

    foo = Foo(a=1, b=2)
    bar = Bar(c=3, d=4)

    baz = Baz.from_dict({"a": 1, "b": 2})
    assert baz == foo
    assert type(baz) == Foo

    bar_json = bar.to_json()
    baz_reverse = Baz.from_json(bar_json)
    assert baz_reverse == bar
    assert type(baz_reverse) == Bar


def test_dataclass_serialize_registry_decorator():
    @serializable_base
    @serializable
    @dataclass
    class Foo:
        a: int
        b: int

    @dataclass
    class Bar(Foo):
        c: int

    foo = Foo(a=1, b=2)
    bar = Bar(a=1, b=2, c=3)

    foo_json = foo.to_json()

    with pytest.raises(TypeError):
        Bar.from_json(foo_json)

    bar_json = bar.to_json()
    foo_reverse = Foo.from_json(bar_json)
    assert foo_reverse == bar
    assert type(foo_reverse) == Bar
