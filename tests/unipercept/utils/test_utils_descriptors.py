from __future__ import annotations

import pytest
from unipercept.utils.descriptors import immutable, objectmagic


def test_objectmagic():
    class Foo(dict):
        @objectmagic
        def __or__(a, b):
            return type(a)({**a, **b})

    a = Foo(x=1)
    b = Foo(y=2)
    c = a | {"y": 2}
    d = a | b

    assert c == {"x": 1, "y": 2}
    assert d == c
    assert Foo.__or__(a, b) == c


def test_immutable():
    class Foo:
        imm = immutable(123)

    foo = Foo()
    with pytest.raises(AttributeError):
        foo.imm = 456
    assert foo.imm == 123
