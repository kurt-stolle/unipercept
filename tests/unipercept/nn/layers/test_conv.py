from __future__ import annotations

import typing as T

import pytest
import torch
import torch.nn as nn
import typing_extensions as TX
from hypothesis import given
from hypothesis import strategies as st

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


num_channels = st.integers(min_value=1, max_value=6)
input_shape = st.tuples(
    st.integers(min_value=6, max_value=16), st.integers(min_value=6, max_value=16)
)
kernel_size = st.sampled_from([3, 5])
stride = st.integers(min_value=1, max_value=2)
padding = st.one_of(st.integers(min_value=0, max_value=1), st.just("same"))


@given(
    num_channels=num_channels,
    input_shape=input_shape,
    kernel_size=kernel_size,
    stride=stride,
    padding=padding,
)
def test_forward(
    conv_module,
    num_channels,
    input_shape,
    kernel_size,
    stride,
    padding,
    activation_module,
    norm_module,
):
    print(f"\n-- {conv_module.__name__} --")
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

    print(
        f"x     : mean {input_tensor.mean().item(): 4.3f}, std {input_tensor.std().item(): 4.3f} | {tuple(input_tensor.shape)}"
    )
    print(
        f"y     : mean {y.mean().item(): 4.3f}, std {y.std().item(): 4.3f} | {tuple(y.shape)}"
    )
    print(
        f"dx/dy : mean {input_tensor.grad.mean().item(): 4.3f}, std {input_tensor.grad.std().item(): 4.3f} | {tuple(input_tensor.grad.shape)}"
    )
