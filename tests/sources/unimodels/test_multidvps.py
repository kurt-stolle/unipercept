import pytest
import torch
import torch.nn as nn
import unimodels.multidvps as multidvps

import unipercept.nn.layers as layers


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("deform", [True, False])
@pytest.mark.parametrize("coord", [None, layers.CoordCat2d])
@pytest.mark.parametrize("num_convs", [1, 3, 5])
def test_multidvps_encoder(batch_size, deform, coord, num_convs):
    in_channels = 32
    out_channels = 32

    encoder = multidvps.modules.Encoder(
        in_channels=in_channels,
        out_channels=out_channels,
        num_convs=num_convs,
        deform=deform,
        coord=coord,
        norm=nn.BatchNorm2d,
        activation=nn.SiLU,
    )
    assert encoder is not None

    inputs = torch.randn(batch_size, in_channels, 64, 64)
    outputs = encoder(inputs)

    assert outputs.shape == (batch_size, out_channels, 64, 64)

    loss = outputs.sum()
    assert loss is not None
    assert all(torch.isfinite(loss))
    loss.backward()

    grad = inputs.grad
    assert grad is not None

    print("\n")
    print(encoder)
    print(f"Inputs   : {inputs.shape}")
    print(f"Outputs  : {outputs.shape}")
    print(f"Loss     : {loss.shape}")
    print(f"Gradient : {grad.shape}")
    print("\n")
