from __future__ import annotations

import pytest
import torch
from unipercept.nn.smooth import EMA, GMA


@pytest.fixture()
def ema():
    return EMA(alpha=0.5)


@pytest.fixture()
def gma():
    return GMA(window_size=5, std_dev=1.0)


@pytest.fixture()
def inputs():
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


def _monotonic_forward(module, inputs):
    for n, i in enumerate(inputs):
        o = module(i)

        if n == 0:
            assert o == i
        else:
            assert o < i

        print(f"{module.__class__.__name__}: {i.item():.2f}, {o.item():.2f}")


def test_ema(ema, inputs):
    _monotonic_forward(ema, inputs)


def test_gma(gma, inputs):
    _monotonic_forward(gma, inputs)
