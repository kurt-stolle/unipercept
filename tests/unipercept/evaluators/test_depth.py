r"""
Tests for `unipercept.evaluators.depth`.
"""

from __future__ import annotations

import pprint
import subprocess
from pathlib import Path

import numpy as np
import pytest
import regex as re
import torch
from torch.utils._pytree import tree_map
from unipercept import file_io
from unipercept.config.env import get_env
from unipercept.data.tensors import DepthFormat, DepthMap
from unipercept.evaluators.depth import EigenMetrics, compute_eigen_metrics
from unipercept.log import create_table

_ABS_TOL = 1e-1


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
        true = torch.tensor([TRUE_DEPTH])
        pred = torch.tensor([PRED_NOISE]) * noise_gain + true
    metrics = compute_eigen_metrics(pred=pred, true=true)
    metrics = tree_map(lambda x: x.cpu().numpy(force=True), metrics)
    pred = pred.numpy(force=True).astype(np.float32)
    true = true.numpy(force=True).astype(np.float32)
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

    assert metrics is not None
    assert pytest.approx(metrics.abs_rel, abs=_ABS_TOL) == float(expected_abs_rel)
    assert pytest.approx(metrics.sq_rel, abs=_ABS_TOL) == float(expected_sq_rel)
    assert pytest.approx(metrics.rmse, abs=_ABS_TOL) == float(expected_rmse)
    assert pytest.approx(metrics.rmse_log, abs=_ABS_TOL) == float(expected_rmse_log)


def test_compute_eigen_metrics_accuracy():
    r"""
    Tests the accuracy metric at canonical thresholds $1.25^n$ for $n \in \{1, 2, 3\}$.
    """
    # Test with a specific scenario where accuracy at different thresholds can be calculated
    pred = torch.tensor([[1.0, 1.3, 1.6]])
    true = torch.tensor([[1.0, 1.0, 1.0]])

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


def _compute_eigen_reference(true_dir: Path, pred_dir: Path) -> EigenMetrics:
    # Compute metrics using reference implementation
    ref_env = "KITTI_DEPTH_DEVKIT_PATH"
    ref_path = get_env(str, ref_env, default="//datasets/kitti-depth/devkit")
    assert ref_path is not None, f"Environment variable {ref_env} not set"
    ref_cli = file_io.Path(ref_path) / "cpp" / "evaluate_depth"

    if not ref_cli.is_file():
        return pytest.skip(f"Reference implementation CLI not found at {ref_cli}")

    ref_cmd = [
        str(ref_cli),
        str(true_dir),
        str(pred_dir),
    ]
    res = subprocess.run(ref_cmd, check=False)

    assert res.returncode == 0, res

    res_pattern = re.compile(R"^mean *([\w ]+): ([\d\.]+).*$")
    ref_metrics = {}

    with open(pred_dir / "stats_depth.txt") as f:
        while True:
            line = f.readline()
            if not line:
                break
            match = res_pattern.match(line)
            if not match:
                continue
            key, value = match.groups()
            ref_metrics[key] = float(value)

    try:
        return EigenMetrics(
            valid=torch.as_tensor(-1),
            accuracy={},
            abs_rel=ref_metrics["abs relative"],
            sq_rel=ref_metrics["squared relative"],
            rmse=ref_metrics["rmse"],
            rmse_log=ref_metrics["log rmse"],
        )
    except KeyError as e:
        msg = f"Reference metrics missing key {e}. Available: {ref_metrics.keys()}"
        raise ValueError(msg) from e


def _check_ours_matches_ref(our_metrics, ref_metrics):
    our_metrics = our_metrics._asdict()
    our_metrics = tree_map(lambda x: x.detach().cpu().item(), our_metrics)
    ref_metrics = ref_metrics._asdict()

    for d in (our_metrics, ref_metrics):
        d.pop("accuracy")
        d.pop("valid")

    diff_dict = {
        key: f"{our_metrics[key]:.6f} vs {ref_metrics[key]:.6f} ({abs(our_metrics[key] - ref_metrics[key]):.3e})"
        for key in our_metrics
    }
    print(f"Metrics: \n{create_table(diff_dict)}")

    for key in our_metrics:
        ours = our_metrics[key]
        refs = ref_metrics[key]
        assert pytest.approx(ours, abs=_ABS_TOL) == refs


@pytest.mark.parametrize("device", ["cuda", "cpu"])
@pytest.mark.parametrize(
    "sample_ids",
    [
        ["000000_000000"],
        ["000000_000001"],
        ["000000_000000", "000000_000001"],
        [
            "000000_000000",
            "000000_000001",
            "000000_000002",
            "000000_000003",
            "000000_000004",
            "000000_000005",
        ],
    ],
)
def test_compute_eigen_metrics_reference_on_sample(device, sample_ids, tmp_path):
    root = (
        file_io.Path(__file__).parent.parent.parent.parent
        / "assets"
        / "testing"
        / "dvps"
    )

    assert root.is_dir(), f"Test data not found at {root} ({root.resolve()})"

    with torch.device(device):
        true = torch.stack(
            [
                DepthMap.read(
                    root / "true" / f"{sample_id}_depth.png",
                    format=DepthFormat.DEPTH_INT16,
                )
                for sample_id in sample_ids
            ]
        )
        pred = torch.stack(
            [
                DepthMap.read(
                    root / "pred" / "depth" / f"{sample_id}.png",
                    format=DepthFormat.DEPTH_INT16,
                )
                for sample_id in sample_ids
            ]
        )

    # Create directories for the true and predicted depth maps
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    for i, item in enumerate(pred):
        item.as_subclass(DepthMap).save(
            pred_dir / f"{i:04d}.png", format=DepthFormat.DEPTH_INT16
        )

    true_dir = tmp_path / "true"
    true_dir.mkdir()
    for i, item in enumerate(true):
        item.as_subclass(DepthMap).save(
            true_dir / f"{i:04d}.png", format=DepthFormat.DEPTH_INT16
        )

    # Compute metrics using the reference implementation and ours
    our_metrics = compute_eigen_metrics(pred=pred, true=true)
    ref_metrics = _compute_eigen_reference(true_dir, pred_dir)

    _check_ours_matches_ref(our_metrics, ref_metrics)


def test_compute_eigen_metrics_reference_on_random(tmp_path):
    r"""
    Tests the Eigen metrics against the KITTI depth reference implementation.
    """

    # Generate fake data
    N, H, W = 10, 64, 32
    MIN_DEPTH = 1.0
    MAX_DEPTH = 100.0
    true_list = [torch.rand(H, W, dtype=torch.float32) * MAX_DEPTH for _ in range(N)]
    pred_list = [true_list[0] + torch.zeros_like(true_list[0])]
    pred_list.extend(
        [
            torch.rand(H, W) * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH
            for _ in range(N - len(pred_list))
        ]
    )

    # Create directories for the true and predicted depth maps
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()

    true_dir = tmp_path / "true"
    true_dir.mkdir()

    for path, tensors in ((pred_dir, pred_list), (true_dir, true_list)):
        for i, tensor in enumerate(tensors):
            dmap = DepthMap((tensor * 256).int() / 256)
            dmap.save(path / f"{i:03d}.png", format=DepthFormat.DEPTH_INT16)

    # Compute metrics using our implemetation
    our_result = compute_eigen_metrics(
        pred=torch.stack(pred_list), true=torch.stack(true_list)
    )
    ref_result = _compute_eigen_reference(true_dir, pred_dir)
    _check_ours_matches_ref(our_result, ref_result)
