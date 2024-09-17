from __future__ import annotations

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import nn
from unipercept.nn.layers import conv


@pytest.fixture(
    scope="module",
    params=[conv.Conv2d, conv.Standard2d, conv.Separable2d, conv.ModDeform2d],
)
def conv_module(request) -> nn.Module:
    return request.param


@pytest.fixture(scope="module", params=[None, nn.ReLU, nn.GELU])
def activation_module(request) -> nn.Module:
    return request.param


@pytest.fixture(scope="module", params=[None, nn.BatchNorm2d])
def norm_module(request) -> nn.Module:
    return request.param


num_channels = st.integers(min_value=1, max_value=2)
kernel_size = st.sampled_from([3, 5])
stride = st.integers(min_value=1, max_value=2)
padding = st.integers(min_value=0, max_value=1)


@given(
    num_channels=num_channels,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
)
def test_forward(
    conv_module,
    num_channels,
    kernel_size,
    stride,
    padding,
    activation_module,
    norm_module,
):
    input_shape = (8, 8)
    input_tensor = torch.randn((2, num_channels, *input_shape), requires_grad=True)
    m = conv_module.with_norm_activation(
        in_channels=num_channels,
        out_channels=3,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        norm=norm_module,
        activation=activation_module,
    )

    y = m.forward(input_tensor)
    out = y.sum()
    assert out.isfinite().all(), out
    out.backward()
    assert input_tensor.grad is not None
