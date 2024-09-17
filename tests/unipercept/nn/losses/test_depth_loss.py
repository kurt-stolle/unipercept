from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from unipercept.nn.losses import depth as depth_losses

torch.autograd.set_detect_anomaly(True)

DEVICES = ["cpu", "cuda"]
MAX_DEPTH = st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False)
INPUTS = array_shapes(min_dims=3, max_dims=3, min_side=4, max_side=16).flatmap(
    lambda shape: st.tuples(
        arrays(
            np.float32,
            shape,
            elements=st.floats(0, 1, width=32, allow_nan=False, allow_infinity=False),
        ),
        arrays(
            np.float32,
            shape,
            elements=st.floats(0, 1, width=32, allow_nan=False, allow_infinity=False),
        ),
        arrays(
            np.bool_,
            shape,
            elements=st.booleans(),
        ),
    )
)


@pytest.mark.parametrize("device", DEVICES)
@settings(deadline=None)
@given(max_depth=MAX_DEPTH, inputs=INPUTS)
def test_depth_loss_sile(
    device: torch.device,
    max_depth: float,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    true_np, pred_np, mask_np = inputs

    true = torch.as_tensor(true_np, dtype=torch.float32, device=device)
    pred = torch.tensor(
        pred_np.tolist(),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    valid_mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device) * (true > 0)
    true = true[valid_mask] * max_depth
    pred = pred[valid_mask] * max_depth
    pred.retain_grad()

    # Calculate loss
    loss = depth_losses.compute_silog_loss(pred, true, valid_mask, dim=(-1,))
    loss = loss.sum()
    assert torch.isfinite(loss).all(), loss

    # Calculate gradients
    loss.backward()

    assert pred.grad is not None, loss
    assert torch.isfinite(pred.grad).all(), pred.grad


@pytest.mark.parametrize("device", DEVICES)
@settings(deadline=None)
@given(max_depth=MAX_DEPTH, inputs=INPUTS)
def test_depth_loss_rel(
    device: torch.device,
    max_depth: float,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    true_np, pred_np, mask_np = inputs

    true = torch.as_tensor(true_np, dtype=torch.float32, device=device)
    pred = torch.tensor(
        pred_np.tolist(),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    valid_mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device) * (true > 0)
    true = true[valid_mask] * max_depth
    pred = pred[valid_mask] * max_depth
    pred.retain_grad()

    # Compute ARE
    loss = depth_losses.compute_relative_loss(pred, true, valid_mask, dim=(-1,))
    loss = loss.sum()
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), pred.grad
