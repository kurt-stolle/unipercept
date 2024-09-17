from __future__ import annotations

import pytest
from unipercept.utils.frozendict import frozendict


def test_frozendict():
    a = frozendict({"x": 1})
    b = frozendict({"y": 2})
    c = a | {"y": 2}
    d = a | b

    assert c == {"x": 1, "y": 2}
    assert d == c

    with pytest.raises(TypeError):
        a["x"] = 1
