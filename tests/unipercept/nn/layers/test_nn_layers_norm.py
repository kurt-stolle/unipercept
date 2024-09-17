from __future__ import annotations

import pytest
import torch
import unipercept.nn.layers.norm as norm_module
from hypothesis import given
from hypothesis import strategies as st

EPS = 1e-8

batch_size = st.integers(min_value=1, max_value=4)
num_channels = st.integers(min_value=1, max_value=16)
input_shape = st.tuples(
    st.integers(min_value=1, max_value=16),
    st.integers(min_value=1, max_value=16),
    st.integers(min_value=1, max_value=16),
)


@given(batch_size=batch_size, num_channels=num_channels, input_shape=input_shape)
def test_layer_norm_chw(batch_size, num_channels, input_shape):
    x = torch.randn((batch_size, num_channels, *input_shape))
    w = torch.randn(num_channels)
    b = torch.randn(num_channels)

    y = norm_module.layer_norm_chw(x, w, b, EPS)

    assert y.shape == x.shape


@pytest.mark.parametrize("channels_last", [True, False])
@given(batch_size=batch_size, num_channels=num_channels, input_shape=input_shape)
def test_global_response_norm(channels_last, batch_size, num_channels, input_shape):
    if channels_last:
        spatial_dim = (1, 2)
        channel_dim = -1
        wb_shape = (1, 1, 1, -1)

        x = torch.randn((batch_size, *input_shape, num_channels))
    else:
        spatial_dim = (2, 3)
        channel_dim = 1
        wb_shape = (1, -1, 1, 1)

        x = torch.randn((batch_size, num_channels, *input_shape))

    w = torch.randn(num_channels)
    b = torch.randn(num_channels)

    y = norm_module.global_response_norm(
        x, spatial_dim, channel_dim, w.view(*wb_shape), b.view(*wb_shape), EPS
    )

    assert y.shape == x.shape
