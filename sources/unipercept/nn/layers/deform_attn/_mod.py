r"""
Multi-Scale Deformable Attention modules
"""

from __future__ import annotations

import math
import typing as T

import torch
import typing_extensions as TX
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.functional import softmax
from torch.nn.init import constant_, xavier_uniform_

from .extension import deform_attn_backward, deform_attn_forward
from .reference import deform_attn as deform_attn_fallback

__all__ = ["MultiScaleDeformAttn", "MultiScaleDeformAttnFunction"]


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        msg = f"Expected a positive integer. Got: {n} ({type(n)})!"
        raise ValueError(msg)
    return (n & (n - 1) == 0) and n != 0


class MultiScaleDeformAttnFunction(Function):
    r"""
    Autograd function for Multi-Scale Deformable Attention.
    """

    @staticmethod
    @TX.override
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @TX.override
    @once_differentiable
    def backward(
        ctx, grad_output: Tensor
    ) -> T.Tuple[Tensor, None, None, Tensor, Tensor, None]:
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformAttn(nn.Module):
    def __init__(self, dims=256, levels=4, heads=8, points=4):
        """
        Multi-Scale Deformable Attention

        Parameters
        ----------
        dims
            amount of hidden dimension in the model
        levels
            number of feature levels
        heads
            number of attention heads
        n_points
            number of sampling points per attention head per feature level
        """
        super().__init__()
        if dims % heads != 0:
            msg = "dims must be divisible by heads, but got {} and {}".format(
                dims, heads
            )
            raise ValueError(msg)
        _d_per_head = dims // heads
        if not _is_power_of_2(_d_per_head):
            msg = "dims / heads must be power of 2, but got {}.".format(_d_per_head)
            raise ValueError(msg)

        self.im2col_step = 128

        self.dims = dims
        self.levels = levels
        self.heads = heads
        self.n_points = points

        self.sampling_offsets = nn.Linear(dims, heads * levels * points * 2)
        self.attention_weights = nn.Linear(dims, heads * levels * points)
        self.value_proj = nn.Linear(dims, dims)
        self.output_proj = nn.Linear(dims, dims)

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
            .repeat(1, self.levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    @TX.override
    def forward(
        self,
        q,
        p,
        v,
        shapes,
        level_index,
        padding_mask=None,
    ):
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
        output : Tensor[N, Q, C]
            The output tensor.
        """
        N, Len_q, _ = q.shape
        N, Len_in, _ = v.shape
        assert (shapes[:, 0] * shapes[:, 1]).sum() == Len_in

        v = self.value_proj(v)
        if padding_mask is not None:
            v = v.masked_fill(padding_mask[..., None], float(0))
        v = v.view(N, Len_in, self.heads, self.dims // self.heads)
        sampling_offsets = self.sampling_offsets(q).view(
            N, Len_q, self.heads, self.levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(q).view(
            N, Len_q, self.heads, self.levels * self.n_points
        )
        attention_weights = softmax(attention_weights, -1).view(
            N, Len_q, self.heads, self.levels, self.n_points
        )
        # N, Len_q, heads, levels, points, 2
        if p.shape[-1] == 2:
            offset_normalizer = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
            sampling_locations = (
                p[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif p.shape[-1] == 4:
            sampling_locations = (
                p[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * p[:, :, None, :, None, 2:] * 0.5
            )
        else:
            msg = f"Last dim of points must be 2 or 4, but get {p.shape[-1]} instead."
            raise ValueError(msg)
        if torch.cuda.is_available():
            out = MultiScaleDeformAttnFunction.apply(
                v,
                shapes,
                level_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            out = deform_attn_fallback(v, shapes, sampling_locations, attention_weights)
        return self.output_proj(out)

    if T.TYPE_CHECKING:
        __call__ = forward
