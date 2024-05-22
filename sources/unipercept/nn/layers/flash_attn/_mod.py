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

__all__ = ["FlashAttn", "FlashAttnFunction"]


def factors(N):
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def findspec(B, Q, G, C):
    d_stride = 8
    ms = factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


def findspec_bwd(B, Q, G, C):
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


class FlashAttnFunction(Function):
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
        d_stride, blockthread = findspec(
            value.shape[0], sampling_loc_attn.shape[1], value.shape[2], value.shape[3]
        )
        d_stride_backward, blockthread_backward = findspec_bwd(
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


class FlashAttn(nn.Module):
    r"""
    Multi-Scale Flash Attention Module
    """

    def __init__(self, channels=256, *, levels=4, heads=8, points=4, **kwargs):
        r"""
        Parameters
        ----------
        channels
            Amount of hidden dimension
        levels
            Amount of feature levels
        heads
            Amount of attention heads
        points
            Amount of sampling points per attention head per feature level
        """
        super().__init__(**kwargs)

        if channels % heads != 0:
            msg = f"{channels=} must be divisible by {heads=}."
            raise ValueError(msg)

        _d_per_head = channels // heads
        assert _is_power_of_2(_d_per_head), (channels, heads)

        self.im2col_step = 64

        self.d_model = channels
        self.n_levels = levels
        self.n_heads = heads
        self.n_points = points

        self.sampling_offsets = nn.Linear(channels, heads * levels * points * 2)
        self.attention_weights = nn.Linear(channels, heads * levels * points)
        self.value_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1).clone())
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    @TX.override
    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        query: Tensor[N, Q, C]
            Query tensor
        reference_points: Tensor[N, Q, L, 2] | Tensor[N, Q, L, 4]
            Reference points for each query point, in the format of (top-left, bottom-right) or (top-left, h, w)
        input_flatten: Tensor[N, H*W, C]
            Flattened input tensor
        input_spatial_shapes: Tensor[L, 2]
            Spatial shapes of each level
        input_level_start_index: Tensor[L]
            Start index of each level in the flattened input tensor
        input_padding_mask: Tensor[N, H*W]
            Padding mask for the input tensor

        Returns
        -------
        Tensor[N, Q, C]
            Output tensor
        """
        N, Q, _ = query.shape
        N, HW, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == HW

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, HW, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Q, self.n_heads, self.n_levels * self.n_points
        )
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            msg = f"Last dim of reference_points must be 2 or 4. Got: {reference_points.shape[-1]}"
            raise ValueError(msg)

        sampling_locations = sampling_locations.flatten(-3).half()
        sampling_loc_attn = torch.cat([sampling_locations, attention_weights], dim=-1)
        output = FlashAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_loc_attn,
            self.im2col_step,
            self.n_points,
        )
        output = self.output_proj(output)
        return output
