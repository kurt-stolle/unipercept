from __future__ import annotations

import typing as T

import pytest
import torch
import typing_extensions as TX

from unipercept.nn.losses import PGTLoss


@pytest.fixture()
def pgt_loss():
    return PGTLoss()


def test_pgt_loss(pgt_loss):
    dep_feat = torch.randn(4, 16, 32, 16)
    seg_true = torch.randint(-1, 10, (4, 32, 16))

    loss = pgt_loss(dep_feat, seg_true)

    print("PGT loss: ", loss.item())

    assert loss >= 0, loss.item()
    assert loss.isfinite(), loss.item()
