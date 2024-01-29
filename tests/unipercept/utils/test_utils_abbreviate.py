from __future__ import annotations

import typing as T

import pytest
import torch.nn as nn
import typing_extensions as TX

from unipercept.utils.abbreviate import full_name, short_name

ABBR_LEN = 4


@pytest.mark.unit
@pytest.mark.parametrize(
    "cls, abr",
    [
        (nn.Conv2d, "co2d"),
        (nn.ConvTranspose2d, "cot2"),
        (nn.Linear, "linr"),
        (nn.Sigmoid, "sigm"),
        (bool, "bool"),
        (int, "int_"),
        (float, "flot"),
    ],
)
def test_short_name(cls, abr):
    full = full_name(cls)
    assert full == cls.__name__, f"Expected {full} to be {cls.__name__}"
    short = short_name(cls, num=len(abr))
    assert short == abr, f"Expected {full} to abbreviate to {abr}, got {short}!"
