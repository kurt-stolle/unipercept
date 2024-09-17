from __future__ import annotations

from unipercept import config


class TestObject:
    def __init__(self, foo, bar):
        self.foo = foo
        self.bar = bar + 1


def test_config_call():
    lazy_obj = config.call(TestObject)(foo=1, bar=2)
    assert lazy_obj.foo == 1
    assert lazy_obj.bar == 2
    assert config.LAZY_TARGET in lazy_obj

    obj = config.instantiate(lazy_obj)
    assert obj.foo == 1
    assert obj.bar == 3
