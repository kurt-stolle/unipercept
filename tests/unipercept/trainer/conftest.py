from __future__ import annotations

import pytest
import torch.nn as nn


@pytest.fixture(scope="session")
def model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, 1, bias=True),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, 1, 1, bias=False),
        nn.BatchNorm2d(32),
        nn.AdaptiveMaxPool2d(1),
    )
