from __future__ import annotations

import torch

from unipercept.nn.point_rend import (
    point_sample,
    random_points,
    random_points_with_importance,
)


def test_point_sample():
    src = torch.arange(4).view(1, 1, 2, 2).float()
    points = torch.tensor([[0.25, 0.25], [0.75, 0.75]]).unsqueeze(0).float()
    samples = point_sample(src, points, mode="nearest", align_corners=False)

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

        samples = point_sample(
            mask.float(), points, mode="nearest", align_corners=False
        )

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
    samples = point_sample(src, points, mode="nearest", align_corners=False)

    assert points.shape == (1, n_points, d_coord)
    assert samples.shape == (1, 1, n_points)
    assert samples.min() >= 3.0, samples.tolist()
    assert samples.max() <= 5.0, samples.tolist()
