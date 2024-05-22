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
    def __init__(self, channels=256, levels=4, heads=8, points=4):
        """
        Multi-Scale Deformable Attention

        Parameters
        ----------
        d_model
            amount of hidden dimension in the model
        levels
            number of feature levels
        heads
            number of attention heads
        n_points
            number of sampling points per attention head per feature level
        """
        super().__init__()
        if channels % heads != 0:
            msg = "d_model must be divisible by heads, but got {} and {}".format(
                channels, heads
            )
            raise ValueError(msg)
        _d_per_head = channels // heads
        if not _is_power_of_2(_d_per_head):
            msg = "d_model / heads must be power of 2, but got {}.".format(_d_per_head)
            raise ValueError(msg)

        self.im2col_step = 128

        self.d_model = channels
        self.levels = levels
        self.heads = heads
        self.n_points = points

        self.sampling_offsets = nn.Linear(channels, heads * levels * points * 2)
        self.attention_weights = nn.Linear(channels, heads * levels * points)
        self.value_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)

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
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.heads, self.d_model // self.heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.heads, self.levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.heads, self.levels * self.n_points
        )
        attention_weights = softmax(attention_weights, -1).view(
            N, Len_q, self.heads, self.levels, self.n_points
        )
        # N, Len_q, heads, levels, points, 2
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
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(
                    reference_points.shape[-1]
                )
            )
        if torch.cuda.is_available():
            output = MultiScaleDeformAttnFunction.apply(
                value,
                input_spatial_shapes,
                input_level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            ## CPU
            output = deform_attn_fallback(
                value, input_spatial_shapes, sampling_locations, attention_weights
            )
        output = self.output_proj(output)
        return output
