r"""
Tests for `unipercept.evaluators.depth`.
"""

from __future__ import annotations

import pprint

import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_map

from unipercept.evaluators.depth import compute_eigen_metrics

TRUE_DEPTH = [2.0**i for i in range(100)]
PRED_NOISE = [((1000.0 / 3) ** (i + 1) % 100) / 100 for i in range(100)]


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize("noise_gain", [1 * (10**n) for n in range(-4, 4)])
def test_compute_eigen_metrics(device: str, noise_gain: float):
    r"""
    Tests the depth metrics computation using the Eigen et al. (2014) formulation.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Sample true and predicted depth values
    with torch.no_grad(), torch.device(device):
        true = torch.tensor(TRUE_DEPTH)
        pred = torch.tensor(PRED_NOISE) * noise_gain + true
    metrics = compute_eigen_metrics(pred=pred, true=true)
    metrics = tree_map(lambda x: x.cpu().numpy(force=True), metrics)
    pred = pred.numpy(force=True)
    true = true.numpy(force=True)
    expected_abs_rel = np.mean(np.abs(true - pred) / true)
    expected_sq_rel = np.mean(((true - pred) ** 2) / true)
    expected_rmse = np.sqrt(np.mean((true - pred) ** 2))
    expected_rmse_log = np.sqrt(np.mean((np.log1p(true) - np.log1p(pred)) ** 2))

    def format_metric(mt, mp):
        diff = np.abs(mt - mp)
        return f"{mt:.6f} vs {mp:.6f} ({diff:.3e})"

    show_metrics: dict[str, str] = {
        "abs_rel": format_metric(expected_abs_rel, metrics.abs_rel),
        "sq_rel": format_metric(expected_sq_rel, metrics.sq_rel),
        "rmse": format_metric(expected_rmse, metrics.rmse),
        "rmse_log": format_metric(expected_rmse_log, metrics.rmse_log),
    }

    print(f"Metrics: \n{pprint.pformat(show_metrics)}")

    rel = 1e-8
    assert metrics is not None
    assert pytest.approx(metrics.abs_rel, rel) == expected_abs_rel
    assert pytest.approx(metrics.sq_rel, rel) == expected_sq_rel
    assert pytest.approx(metrics.rmse, rel) == expected_rmse
    assert pytest.approx(metrics.rmse_log, rel) == expected_rmse_log


def test_compute_eigen_metrics_with_invalid_input():
    r"""
    Tests the depth metrics computation with invalid input.
    """
    pred = torch.tensor([])
    true = torch.tensor([])

    # Ensure the function handles empty inputs gracefully
    metrics = compute_eigen_metrics(pred=pred, true=true)

    # Expect the function to return None for invalid/empty inputs
    assert metrics.valid == 0


def test_compute_eigen_metrics_accuracy():
    r"""
    Tests the accuracy metric at canonical thresholds $1.25^n$ for $n \in \{1, 2, 3\}$.
    """
    # Test with a specific scenario where accuracy at different thresholds can be calculated
    pred = torch.tensor([1.0, 1.3, 1.6])
    true = torch.tensor([1.0, 1.0, 1.0])

    # Calculate metrics
    metrics = compute_eigen_metrics(pred=pred, true=true, t_base=1.25, t_n=[1, 2, 3])

    # Verify accuracy at different thresholds
    accuracy = metrics.accuracy
    assert "1t25**1" in accuracy
    assert "1t25**2" in accuracy
    assert "1t25**3" in accuracy

    # Expected accuracy values for this specific scenario
    expected_accuracy_1 = 1 / 3  # 1/3 within the 1.25^1 threshold
    expected_accuracy_2 = 2 / 3  # 2/3 within the 1.25^2 threshold
    expected_accuracy_3 = 3 / 3  # 3/3 within the 1.25^3 threshold

    # Verify the calculated accuracies match the expected values
    assert pytest.approx(accuracy["1t25**1"], 1e-4) == expected_accuracy_1
    assert pytest.approx(accuracy["1t25**2"], 1e-4) == expected_accuracy_2
    assert pytest.approx(accuracy["1t25**3"], 1e-4) == expected_accuracy_3
