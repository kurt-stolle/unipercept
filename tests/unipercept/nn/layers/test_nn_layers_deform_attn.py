from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck
from unipercept.nn.layers.deform_attn import (
    MultiScaleDeformAttnFunction,
    MultiScaleFlashAttnFunction,
    reference,
)

N, M, D = 1, 2, 2
Q, L, P = 2, 2, 2


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@torch.no_grad()
def test_deform_attn_reference(dtype: torch.dtype):
    if dtype in (torch.float16, torch.bfloat16):
        pytest.xfail("CUDA implementation does not (yet) support half precision")
    with torch.device("cuda"):
        shapes = torch.as_tensor([(32, 16), (16, 8)], dtype=torch.long)
        level_start_index = torch.cat(
            (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
        )
        S = sum([(H * W).item() for H, W in shapes])

        value = torch.rand(N, S, M, D) * 0.01
        loc = torch.rand(N, Q, M, L, P, 2)
        attn = torch.rand(N, Q, M, L, P) + 1e-5
        attn /= attn.sum(-1, keepdim=True).sum(-2, keepdim=True)
        im2col_step = 8

        value = value.to(dtype=dtype)
        loc = loc.to(dtype=dtype)
        attn = attn.to(dtype=dtype)

        output_pytorch = (
            reference.deform_attn(
                value,
                shapes,
                loc,
                attn,
            )
            .detach()
            .cpu()
        )
        output_cuda = (
            MultiScaleDeformAttnFunction.apply(
                value,
                shapes,
                level_start_index,
                loc.to(dtype=dtype),
                attn.to(dtype=dtype),
                im2col_step,
            )
            .detach()
            .cpu()
        )
        fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
        max_abs_err = (output_cuda - output_pytorch).abs().max()
        max_rel_err = (
            (output_cuda - output_pytorch).abs() / output_pytorch.abs()
        ).max()

    print(max_abs_err, max_rel_err)
    assert fwdok, (max_abs_err, max_rel_err)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@torch.no_grad()
def test_flash_attn_reference(dtype: torch.dtype):
    with torch.device("cuda"):
        pytest.skip()

        shapes = torch.as_tensor([(32, 16), (16, 8)], dtype=torch.long)
        level_idx = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
        S = sum([(H * W).item() for H, W in shapes])

        value = torch.rand(N, S, M, D) * 0.01
        loc = torch.rand(N, Q, M, L, P, 2)
        attn = torch.rand(N, Q, M, L, P) + 1e-5
        attn /= attn.sum(-1, keepdim=True).sum(-2, keepdim=True)
        im2col_step = 8

        value = value.to(dtype=dtype)
        loc = loc.to(dtype=dtype)
        attn = attn.to(dtype=dtype)

        output_pytorch = (
            reference.deform_attn(
                value,
                shapes,
                loc,
                attn,
            )
            .detach()
            .cpu()
        )
        output_cuda = (
            MultiScaleFlashAttnFunction.apply(
                value,
                shapes,
                level_idx,
                torch.cat([loc.flatten(-3), attn.flatten(-2)], dim=-1),
                im2col_step,
            )
            .detach()
            .cpu()
        )
        fwdok = torch.allclose(output_cuda, output_pytorch, rtol=1e-2, atol=1e-3)
        max_abs_err = (output_cuda - output_pytorch).abs().max()
        max_rel_err = (
            (output_cuda - output_pytorch).abs() / output_pytorch.abs()
        ).max()

    print(max_abs_err, max_rel_err)
    assert fwdok, (max_abs_err, max_rel_err)


@pytest.mark.parametrize("channels", [30, 32, 64, 71, 1025, 2048, 3096])
def test_deform_attn_gradient(
    channels, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True
):
    with torch.device("cuda"):
        shapes = torch.as_tensor([(32, 16), (16, 8)], dtype=torch.long)
        level_start_index = torch.cat(
            (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
        )
        S = sum([(H * W).item() for H, W in shapes])

        value = torch.rand(N, S, M, channels) * 0.01
        sampling_locations = torch.rand(N, Q, M, L, P, 2)
        attention_weights = torch.rand(N, Q, M, L, P) + 1e-5
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(
            -2, keepdim=True
        )
        im2col_step = 8
        func = MultiScaleDeformAttnFunction.apply

        value.requires_grad = grad_value
        sampling_locations.requires_grad = grad_sampling_loc
        attention_weights.requires_grad = grad_attn_weight

    assert gradcheck(
        func,
        (
            value.double(),
            shapes,
            level_start_index,
            sampling_locations.double(),
            attention_weights.double(),
            im2col_step,
        ),
    )


@pytest.mark.parametrize("channels", [30, 32, 64, 71, 1025, 2048, 3096])
def test_flash_attn_gradient(
    channels, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True
):
    with torch.device("cuda"):
        shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
        level_start_index = torch.cat(
            (shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1])
        )
        S = sum([(H * W).item() for H, W in shapes])

        value = torch.rand(N, S, M, channels) * 0.01
        loc = torch.rand(N, Q, M, L, P, 2)
        attn = torch.rand(N, Q, M, L, P) + 1e-5
        attn /= attn.sum(-1, keepdim=True).sum(-2, keepdim=True)
        im2col_step = 2
        func = MultiScaleFlashAttnFunction.apply

        value.requires_grad = grad_value
        loc.requires_grad = grad_sampling_loc
        attn.requires_grad = grad_attn_weight

    assert gradcheck(
        func,
        (
            value.double(),
            shapes,
            level_start_index,
            torch.cat([loc.double(), attn.double()], dim=-1),
            im2col_step,
        ),
    )
