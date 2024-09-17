from __future__ import annotations

import torch
from unipercept.nn.layers.ops import dynamic_conv2d


def test_dynamic_conv2d():
    n, d, h, w = (4, 8, 16, 32)
    k = torch.randn(n, d)
    f = torch.randn(d, h, w)
    o = dynamic_conv2d(f, k)

    assert o.shape == (n, h, w)


def test_dynamic_conv2d__batched():
    batch_num = 4
    n, d, h, w = (4, 8, 16, 32)
    ks = [torch.randn(n, d) for _ in range(batch_num)]
    kt = torch.stack(ks)
    fs = [torch.randn(d, h, w) for _ in range(batch_num)]
    ft = torch.stack(fs)

    # Compute the output for each pair of features and kernels
    out_single = torch.stack(
        [dynamic_conv2d(f, k) for f, k in zip(fs, ks, strict=False)]
    )

    # Compute the output for all features and kernels at once
    out_batched = dynamic_conv2d(ft, kt)

    assert torch.allclose(out_single, out_batched)
