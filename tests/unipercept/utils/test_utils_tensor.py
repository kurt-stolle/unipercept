from __future__ import annotations

import pytest
import torch
from unipercept.utils.tensor import map_values


@pytest.mark.parametrize(
    "tensor, translation",
    [
        (
            torch.randint(0, 256, (16, 224, 224, 3)),
            (torch.arange(0, 256), torch.randperm(256)),
        ),
        (
            torch.randint(-128, 256, (16, 224, 224, 3)),
            (torch.arange(0, 128), torch.randperm(128)),
        ),
    ],
)
def test_map_values(tensor, translation):
    result = map_values(tensor, translation, default=translation[1][0])
    assert torch.isin(result, translation[1]).all()
