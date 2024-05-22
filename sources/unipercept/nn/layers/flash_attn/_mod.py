r"""
Multi-scale Deformable Flash Attention modules
"""

from __future__ import annotations

import math
import typing as T

import torch
import typing_extensions as TX
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

from .extension import flash_attn_backward, flash_attn_forward

__all__ = ["MultiScaleFlashAttn", "MultiScaleFlashAttnFunction"]


def factors(N):
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def _lookup_forward_stride_thread(B, Q, G, C):
    d_stride = 8
    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


def _lookup_backward_stride_thread(B, Q, G, C):
    if C >= 64:
        d_stride = 2
    else:
        d_stride = 1

    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


class MultiScaleFlashAttnFunction(Function):
    @staticmethod
    @TX.override
    @torch.autocast("cuda", enabled=True, dtype=torch.float16)
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_loc_attn,
        im2col_step,
        K=8,
    ):
        ctx.im2col_step = im2col_step
        ctx.K = K
        d_stride, blockthread = _lookup_forward_stride_thread(
            value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3]
        )
        d_stride_backward, blockthread_backward = _lookup_backward_stride_thread(
            value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3]
        )

        ctx.d_stride_backward = d_stride_backward
        ctx.blockthread_backward = blockthread_backward

        output = flash_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            ctx.im2col_step,
            K,
            d_stride,
            blockthread,
        )
        ctx.save_for_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_loc_attn
        )
        return output

    @staticmethod
    @TX.override
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc_attn = flash_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_loc_attn,
            grad_output.contiguous(),
            ctx.im2col_step,
            ctx.K,
            ctx.d_stride_backward,
            ctx.blockthread_backward,
        )

        return grad_value, None, None, grad_sampling_loc_attn, None, None


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n))
        )
    return (n & (n - 1) == 0) and n != 0


class MultiScaleFlashAttn(nn.Module):
    r"""
    Multi-Scale Flash Attention Module
    """

    def __init__(
        self,
        dims=256,
        *,
        levels=4,
        heads=8,
        points=4,
        proj: type[nn.Module] = nn.Linear,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        dims
            Amount of hidden dimension
        levels
            Amount of feature levels
        heads
            Amount of attention heads
        points
            Amount of sampling points per attention head per feature level
        """
        super().__init__(**kwargs)

        if dims % heads != 0:
            msg = f"{dims=} must be divisible by {heads=}."
            raise ValueError(msg)

        _d_per_head = dims // heads
        assert _is_power_of_2(_d_per_head), (dims, heads)

        self.im2col_step = 64

        self.dims = dims
        self.levels = levels
        self.heads = heads
        self.points = points

        self.sampling_offsets = nn.Linear(dims, heads * levels * points * 2)
        self.attention_weights = nn.Linear(dims, heads * levels * points)
        self.proj_input = proj(dims, dims)
        self.proj_output = proj(dims, dims)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.heads, dtype=torch.float32) * (
            2.0 * math.pi / self.heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.heads, 1, 1, 2)
            .repeat(1, self.levels, self.points, 1)
        )
        for i in range(self.points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1).clone())
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.proj_input.weight.data)
        constant_(self.proj_input.bias.data, 0.0)
        xavier_uniform_(self.proj_output.weight.data)
        constant_(self.proj_output.bias.data, 0.0)

    @TX.override
    def forward(
        self,
        q: Tensor,
        p: Tensor,
        v: Tensor,
        shapes: Tensor,
        level_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        q: Tensor[N, Q, C]
            Query tensor
        p: Tensor[N, Q, L, 2] | Tensor[N, Q, L, 4]
            Reference points for each query point, in the format of (top-left, bottom-right) or (top, left, h, w)
        v: Tensor[N, H*W, C]
            Flattened input tensor
        shapes: Tensor[L, 2]
            Spatial shapes of each level
        level_index: Tensor[L]
            Start index of each level in the flattened input tensor
        padding_mask: Tensor[N, H*W]
            Padding mask for the input tensor

        Returns
        -------
        Tensor[N, Q, C]
            Output tensor
        """
        N, Q, _ = q.shape
        N, HW, _ = v.shape
        assert (shapes[:, 0] * shapes[:, 1]).sum() == HW

        v = self.proj_input(v)
        if padding_mask is not None:
            v = v.masked_fill(padding_mask[..., None], float(0))
        v = v.view(N, HW, self.heads, self.dims // self.heads)
        off = self.sampling_offsets(q).view(
            N, Q, self.heads, self.levels, self.points, 2
        )
        attn = self.attention_weights(q).view(
            N, Q, self.heads, self.levels * self.points
        )
        # N, Q, heads, levels, points, 2
        if p.shape[-1] == 2:
            off_norm = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
            loc = (
                p[:, :, None, :, None, :] + off / off_norm[None, None, None, :, None, :]
            )
        elif p.shape[-1] == 4:
            loc = (
                p[:, :, None, :, None, :2]
                + off / self.points * p[:, :, None, :, None, 2:] * 0.5
            )
        else:
            msg = f"Last dim of points must be 2 or 4. Got: {p.shape[-1]}"
            raise ValueError(msg)

        loc = loc.flatten(-3).half()
        loc_attn = torch.cat([loc, attn], dim=-1)
        out = MultiScaleFlashAttnFunction.apply(
            v,
            shapes,
            level_index,
            loc_attn,
            self.im2col_step,
            self.points,
        )
        return self.proj_output(out)
