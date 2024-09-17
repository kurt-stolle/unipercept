from __future__ import annotations

from unipercept.utils.decorators import shadowmutate


def test_shadowmethod():
    class Foo(dict):
        @shadowmutate
        def test(self, k, v):
            self[k] = v

            return self

    foo = Foo(x=1)
    shadow_foo = foo.test("x", 2)
    assert foo["x"] == 1
    assert shadow_foo["x"] == 2

    class Bar(dict):
        @shadowmutate()
        def test(self, k, v):
            self[k] = v

            return self

    bar = Bar(x=1)
    shadow_bar = bar.test("x", 2)
    assert bar["x"] == 1
    assert shadow_bar["x"] == 2
