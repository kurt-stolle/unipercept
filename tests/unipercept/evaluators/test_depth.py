r"""
Tests for `unipercept.evaluators.depth`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unipercept.evaluators import compute_depth_metrics


def test_compute_depth_metrics_eigen_etal():
    r"""
    Tests the depth metrics computation using the Eigen et al. (2014) formulation.
    """
    # Sample true and predicted depth values
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    true = torch.tensor([1.0, 2.1, 2.9, 4.0])
    metrics = compute_depth_metrics(pred=pred, true=true)

    pred = pred.numpy(force=True).astype(np.float128)
    true = true.numpy(force=True).astype(np.float128)
    expected_abs_rel = np.mean(np.abs(true - pred) / true)
    expected_sq_rel = np.mean(((true - pred) ** 2) / true)
    expected_rmse = np.sqrt(np.mean((true - pred) ** 2))
    expected_rmse_log = np.sqrt(np.mean((np.log1p(true) - np.log1p(pred)) ** 2))

    rel = 1e-8
    assert metrics is not None
    assert pytest.approx(metrics.abs_rel, rel) == expected_abs_rel
    assert pytest.approx(metrics.sq_rel, rel) == expected_sq_rel
    assert pytest.approx(metrics.rmse, rel) == expected_rmse
    assert pytest.approx(metrics.rmse_log, rel) == expected_rmse_log


def test_compute_depth_metrics_with_invalid_input():
    r"""
    Tests the depth metrics computation with invalid input.
    """
    pred = torch.tensor([])
    true = torch.tensor([])

    # Ensure the function handles empty inputs gracefully
    metrics = compute_depth_metrics(pred=pred, true=true)

    # Expect the function to return None for invalid/empty inputs
    assert metrics.valid == 0


def test_compute_depth_metrics_accuracy():
    r"""
    Tests the accuracy metric at canonical thresholds $1.25^n$ for $n \in \{1, 2, 3\}$.
    """
    # Test with a specific scenario where accuracy at different thresholds can be calculated
    pred = torch.tensor([1.0, 1.3, 1.6])
    true = torch.tensor([1.0, 1.0, 1.0])

    # Calculate metrics
    metrics = compute_depth_metrics(pred=pred, true=true, t_base=1.25, t_n=[1, 2, 3])

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
