from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from unipercept.nn.losses import DepthLoss
from unipercept.nn.losses.functional import (
    relative_absolute_squared_error,
    scale_invariant_logarithmic_error,
)

devices = [torch.device("cpu")]
# if torch.cuda.is_available():
#     devices.append(torch.device("cuda"))

torch.autograd.set_detect_anomaly(True)

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


@pytest.mark.parametrize("device", devices)
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

    if not valid_mask.any():
        return

    true = true[valid_mask] * max_depth
    pred = pred[valid_mask] * max_depth
    pred.retain_grad()

    # Calculate loss
    num = pred.shape[0]
    loss = scale_invariant_logarithmic_error(pred, true, num, eps=1e-8)
    assert torch.isfinite(loss).all(), loss

    # Calculate gradients
    loss.backward()

    assert pred.grad is not None, loss
    assert torch.isfinite(pred.grad).all(), pred.grad


@pytest.mark.parametrize("device", devices)
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

    if not valid_mask.any():
        return

    true = true[valid_mask] * max_depth
    pred = pred[valid_mask] * max_depth
    pred.retain_grad()

    # Calculate loss
    num = pred.shape[0]
    # Compute ARE
    are, sre = relative_absolute_squared_error(pred, true, num, eps=1e-8)
    loss = torch.stack([are, sre]).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all(), pred.grad


@pytest.mark.parametrize("device", devices)
@settings(deadline=None)
@given(max_depth=MAX_DEPTH, inputs=INPUTS)
def test_loss_depth_means(
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
    valid_mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)

    # Calculate loss
    loss_fn = DepthLoss()
    loss = loss_fn(true * max_depth, pred * max_depth, mask=valid_mask)
    loss = loss.sum()

    assert loss is not None
    assert torch.isfinite(loss).all(), loss

    # Calculate gradients
    loss.backward(create_graph=True, retain_graph=True)

    assert pred.grad is not None, loss
    assert torch.isfinite(pred.grad).all(), pred.grad


@pytest.mark.parametrize("device", devices)
@settings(deadline=None)
@given(max_depth=MAX_DEPTH, inputs=INPUTS)
def test_loss_depth_values(
    device: torch.device,
    max_depth: float,
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    true_np, pred_np, valid_np = inputs

    valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device)
    true = torch.as_tensor(true_np, dtype=torch.float32, device=device)
    pred = torch.tensor(
        pred_np.tolist(),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    pred.retain_grad()

    # Calculate loss
    loss_fn = DepthLoss()
    loss = loss_fn(
        true * max_depth,
        pred * max_depth,
        mask=valid,
    )

    assert loss is not None
    assert torch.isfinite(loss), loss

    # Calculate gradients
    loss.backward(retain_graph=True, create_graph=True)

    assert pred.grad is not None, loss
    assert torch.isfinite(pred.grad).all(), pred.grad
