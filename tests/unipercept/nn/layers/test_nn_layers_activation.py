from __future__ import annotations

import pytest
import torch
from unipercept.nn.activations import inverse_softplus, softplus
from unipercept.types import DType


@pytest.mark.parametrize(
    "input",
    [i * 5 for i in range(-10, 10)],
)
@pytest.mark.parametrize(
    "beta",
    [1.0, 10.0, 100.0],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.float16, torch.bfloat16],
)
def test_inverse_softplus(input: float, beta: float, dtype: DType) -> None:
    """
    Test inverse softplus function.
    """
    x = torch.as_tensor(input, dtype=dtype)
    y = softplus(x, beta=beta)
    z = inverse_softplus(y, beta=beta)

    print(f"x = {x.item():.2f}, beta = {beta}")
    print(f"y = softplus({x.item():.2f}, beta={beta}) = {y.item():.2f}")
    print(f"z = inverse_softplus(y, beta={beta}) = {z.item():.2f}")
    print(f"d = |x - z| = {torch.abs(x - z).item():.4e}")

    assert y == 0.0 or torch.allclose(z, x), (x.item(), y.item(), z.item())
