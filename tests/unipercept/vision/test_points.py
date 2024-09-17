from __future__ import annotations

import pytest
import torch
from unipercept.vision.point import (
    bins_by_values,
    map_in_bins,
    random_points,
    random_points_with_bins,
    random_points_with_importance,
    sample,
)


def test_sample():
    src = torch.arange(4).view(1, 1, 2, 2).float()
    points = torch.tensor([[0.25, 0.25], [0.75, 0.75]]).unsqueeze(0).float()
    samples = sample(src, points, mode="nearest", align_corners=False)

    assert samples.shape == (1, 1, points.shape[2])
    assert samples[0, 0, 0] == 0, samples.tolist()
    assert samples[0, 0, 1] == 3, samples.tolist()


def test_random_points():
    src = torch.rand(1, 1, 10, 10)
    n_points = 10
    d_coord = 2
    points = random_points(src, n_points, d_coord)

    assert points.shape == (1, n_points, d_coord)
    assert points.min() >= 0
    assert points.max() <= 1


def test_random_points_with_mask():
    src = torch.rand(1, 1, 100, 100)
    n_points = 5
    d_coord = 2
    for mask in (
        torch.tensor(
            [[False, True, False], [False, False, False], [False, False, False]]
        )[None, None, :, :],
        torch.arange(20).view(1, 1, 2, 10) == 15,
        torch.rand_like(src) < 0.5,  # mild sparsity
        torch.rand_like(src) < 0.1,  # high sparsity
    ):
        points = random_points(src, n_points, d_coord, mask)

        assert points.shape == (1, n_points, d_coord)
        assert points.min() >= 0
        assert points.max() <= 1

        samples = sample(mask.float(), points, mode="nearest", align_corners=False)

        assert samples.shape == (1, 1, n_points)
        assert torch.all(samples.bool()), samples.tolist()


def test_random_points_with_importance():
    src = torch.arange(6).view(1, 1, 2, 3).float()
    n_points = 3
    d_coord = 2

    def importance_fn(t):
        return t

    points = random_points_with_importance(
        src,
        n_points,
        d_coord,
        oversample_ratio=n_points * 100,
        importance_sample_ratio=1.0,
        importance_fn=importance_fn,
        mode="nearest",
        align_corners=False,
    )
    samples = sample(src, points, mode="nearest", align_corners=False)

    assert points.shape == (1, n_points, d_coord)
    assert samples.shape == (1, 1, n_points)
    assert samples.min() >= 3.0, samples.tolist()
    assert samples.max() <= 5.0, samples.tolist()


@pytest.mark.parametrize("use_batch", [False, True])
@pytest.mark.parametrize("mode", ["linear", "quantile"])
def test_bins_by_values(use_batch, mode):
    n_bins = 10
    n_values = 30
    n_batch = 4
    values = torch.rand(n_batch, n_values)
    bins = bins_by_values(values, n_bins, use_batch=use_batch, mode=mode)

    print(
        f"{use_batch=}, {mode=}:\n- values[0] : {values[0].tolist()}\n- bins[0]   : {bins[0].tolist()}",
        flush=True,
    )

    assert bins.shape == (n_batch, n_bins)
    assert bins.min() >= 0
    assert bins.max() <= 1
    # assert torch.all(bins[1:] >= bins[:-1]), bins.tolist()


@pytest.mark.parametrize("use_batch", [False, True])
@pytest.mark.parametrize("mode", ["linear", "quantile"])
def test_map_in_bins(use_batch: bool, mode: str):
    n_bins = 10
    n_values = 1000
    n_batch = 2
    values = torch.rand(n_batch, n_values)
    mask = values > 0.1
    bins = bins_by_values(values, n_bins, mask=mask, use_batch=use_batch, mode=mode)

    def masked_mean(t, m):
        t = t.masked_fill(~m, torch.nan)
        return torch.nanmean(t, dim=1, keepdim=True)

    mapped = map_in_bins(values, bins, masked_mean)

    assert mapped.shape == (n_batch, n_bins)

    print(
        f"{use_batch=}, {mode=}:\n- mapped[0] : {mapped[0].tolist()}\n- mapped[1]: {mapped[1].tolist()}",
        flush=True,
    )


@pytest.mark.parametrize("use_batch", [False, True])
@pytest.mark.parametrize("mode", ["linear", "quantile"])
def test_map_in_bins_with_log_values(use_batch: bool, mode: str):
    n_bins = 10
    n_values = 1000
    n_batch = 2
    sources = torch.rand(n_batch, n_values)
    values = sources.log1p()
    bins = bins_by_values(values, n_bins, use_batch=use_batch, mode=mode)

    def masked_mean(t, m):
        t = t.masked_fill(~m, torch.nan)
        return torch.nanmean(t, dim=1, keepdim=True)

    mapped = map_in_bins(sources, bins, masked_mean, values=values)

    assert mapped.shape == (n_batch, n_bins)

    print(
        f"{use_batch=}, {mode=}:\n- mapped[0] : {mapped[0].tolist()}\n- mapped[1]: {mapped[1].tolist()}",
        flush=True,
    )


@pytest.mark.parametrize("use_batch", [False, True])
@pytest.mark.parametrize("mode", ["linear", "quantile"])
def test_map_in_bins_with_itemcount(use_batch: bool, mode: str):
    n_bins = 10
    n_values = 1000
    n_batch = 2
    sources = torch.rand(n_batch, n_values)
    values = sources.log1p()
    bins = bins_by_values(values, n_bins, use_batch=use_batch, mode=mode)

    def select_count(t, m):
        return m.int().sum(dim=1, keepdim=True)

    counts = map_in_bins(sources, bins, select_count, values=values)
    totals = counts.sum(dim=1)

    assert counts.shape == (n_batch, n_bins)

    print(
        f"{use_batch=}, {mode=}:\n- counts[0] : {counts[0].tolist()}\n- counts[1]: {counts[1].tolist()}\n- total: {totals.tolist()}",
        flush=True,
    )


@pytest.mark.parametrize("use_batch", [False, True])
@pytest.mark.parametrize("mode", ["linear", "quantile"])
def test_map_in_bins_with_itemcount_masked(use_batch: bool, mode: str):
    n_bins = 10
    n_values = 1000
    n_batch = 2
    sources = torch.rand(n_batch, n_values)
    values = sources.log1p()
    mask = sources > 0.1
    bins = bins_by_values(values, n_bins, mask=mask, use_batch=use_batch, mode=mode)

    def select_count(t, m):
        return m.int().sum(dim=1, keepdim=True)

    counts = map_in_bins(sources, bins, select_count, values=values, mask=mask)
    totals = counts.sum(dim=1)

    assert counts.shape == (n_batch, n_bins)

    print(
        f"{use_batch=}, {mode=}:\n- counts[0] : {counts[0].tolist()}\n- counts[1]: {counts[1].tolist()}\n- total: {totals.tolist()}\n- masked: {mask.int().sum(dim=1).tolist()}",
        flush=True,
    )


@pytest.mark.parametrize("use_batch", [False, True])
@pytest.mark.parametrize("mode", ["linear", "quantile"])
def test_random_points_with_bins(use_batch: bool, mode: str):
    n_bins = 10
    n_points = 8
    n_batch = 2
    sources = torch.rand(n_batch, 1, 16, 16)
    values = sources.log1p()

    random_points = random_points_with_bins(
        sources, n_points, use_batch=use_batch, mode=mode, n_bins=n_bins, values=values
    )

    assert random_points.shape == (n_batch, n_points, 2)
