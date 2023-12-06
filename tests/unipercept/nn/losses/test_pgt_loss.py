import pytest
import torch

from unipercept.nn.losses import PGTLoss


@pytest.fixture
def pgt_loss():
    return PGTLoss(p_height=5, p_width=5, margin=0.3)


def test_pgt_loss_dimensions(pgt_loss):
    # Test for correct input dimensions
    output = torch.randn(16, 1, 100, 100)
    target = torch.randint(0, 2, (16, 100, 100), dtype=torch.float32)

    loss = pgt_loss(output, target)
    assert loss.dim() == 0, f"Expected scalar output, got dimensions: {loss.dim()}"


def test_pgt_loss_zeros(pgt_loss):
    # Test for all zero target tensor
    output = torch.randn(16, 1, 100, 100)
    target = torch.zeros((16, 100, 100), dtype=torch.float32)

    loss = pgt_loss(output, target)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7), f"Expected loss to be close to 0, got {loss.item()}"


def test_pgt_loss_values(pgt_loss):
    # Test for non-zero loss when positive and negative patches exist
    output = torch.randn(16, 1, 100, 100)
    target = torch.cat(
        [
            torch.randn((8, 100, 100), dtype=torch.float32),
            torch.ones((8, 100, 100), dtype=torch.float32),
        ]
    )

    loss = pgt_loss(output, target)
    assert loss > 0, f"Expected loss to be greater than 0, got {loss.item()}"


def test_pgt_loss_determinism(pgt_loss):
    # Test for deterministic output given the same input
    output = torch.randn(16, 1, 100, 100)
    target = torch.randint(0, 2, (16, 100, 100), dtype=torch.float32)

    loss1 = pgt_loss(output, target)
    loss2 = pgt_loss(output, target)
    assert torch.isclose(
        loss1, loss2, atol=1e-7
    ), f"Expected deterministic behavior, got {loss1.item()} and {loss2.item()}"


def test_pgt_loss_margin(pgt_loss):
    # Test for correct usage of margin parameter
    output = torch.randn(16, 1, 100, 100)
    target = torch.cat(
        [
            torch.zeros((8, 100, 100), dtype=torch.float32),
            torch.ones((8, 100, 100), dtype=torch.float32),
        ]
    )

    pgt_loss_high_margin = PGTLoss(p_height=5, p_width=5, margin=1.0)
    loss_high = pgt_loss_high_margin(output, target)
    loss = pgt_loss(output, target)

    assert loss_high >= loss, f"Expected higher loss with greater margin, got {loss_high.item()} and {loss.item()}"
