import pytest
import torch

from unipercept.nn import layers


@pytest.mark.unit
def test_projection_field():
    key_location = "mask"
    key_depth = "depth"

    value_field = layers.projection.DepthProjection(max_depth=100.0)

    data = {
        key_location: torch.arange(3 * 3).reshape(3, 3).bool()[None, :],
        key_depth: torch.arange(3).float()[None, :],
    }

    assert data[key_location].ndim == 3
    assert data[key_depth].ndim == 2

    value_result = value_field(data, {})

    assert value_result is not None
