import pytest
import torch
from unipercept.nn.layers.dynamic_conv import (
    _dynamic_conv2d_einsum,
    _dynamic_conv2d_matmul,
    _dynamic_conv2d_naive,
)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_dynamic_conv2d(dtype, device):
    for n in range(100):
        f = torch.randn(4, 8, 16, 32, dtype=dtype, device=device)
        k = torch.randn(4, 10, 8, dtype=dtype, device=device)

        out1 = _dynamic_conv2d_naive(f, k)
        out2 = _dynamic_conv2d_einsum(f, k)
        out3 = _dynamic_conv2d_matmul(f, k)

        assert torch.allclose(out1, out2)
        assert torch.allclose(out1, out3)
        assert torch.allclose(out2, out3)
