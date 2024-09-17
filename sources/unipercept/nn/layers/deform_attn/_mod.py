r"""
Multi-Scale Deformable Attention modules
"""

import math
import typing as T

import torch
import torch.fx
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_

from .extension import (
    deform_attn_backward,
    deform_attn_forward,
    flash_attn_backward,
    flash_attn_forward,
)
from .reference import deform_attn as deform_attn_native

__all__ = [
    "MultiScaleDeformAttn",
    "MultiScaleDeformAttnFunction",
    "MultiScaleFlashAttn",
    "MultiScaleFlashAttnFunction",
    "MultiScaleNativeAttn",
]


def _get_factors(N):
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        msg = f"Expected a positive integer. Got: {n} ({type(n)})!"
        raise ValueError(msg)
    return (n & (n - 1) == 0) and n != 0


def _is_divisible(a, b):
    return a % b == 0


def _lookup_forward_stride_thread(B, Q, G, C):
    d_stride = 8
    ms = _get_factors(B * Q)
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

    ms = _get_factors(B * Q)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


class MultiScaleFlashAttnFunction(Function):
    @staticmethod
    @torch.compiler.disable
    @T.override
    def forward(
        ctx,
        value,
        spatial_shapes,
        start_index,
        loc_attn,
        im2col_step,
        points=8,
    ):
        ctx.im2col_step = im2col_step
        ctx.points = points
        stride_fwd, blkthd_fwd = _lookup_forward_stride_thread(
            value.shape[0], loc_attn.shape[1], value.shape[2], value.shape[3]
        )
        stride_bwd, blkthd_bwd = _lookup_backward_stride_thread(
            value.shape[0], loc_attn.shape[1], value.shape[2], value.shape[3]
        )
        ctx.stride_bwd = stride_bwd
        ctx.blkthd_bwd = blkthd_bwd

        output = flash_attn_forward(
            value,
            spatial_shapes,
            start_index,
            loc_attn,
            ctx.im2col_step,
            points,
            stride_fwd,
            blkthd_fwd,
        )
        ctx.save_for_backward(value, spatial_shapes, start_index, loc_attn)
        return output

    @staticmethod
    @T.override
    @torch.compiler.disable
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            spatial_shapes,
            level_index,
            loc_attn,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc_attn = flash_attn_backward(
            value,
            spatial_shapes,
            level_index,
            loc_attn,
            grad_output.contiguous(),
            ctx.im2col_step,
            ctx.points,
            ctx.stride_bwd,
            ctx.blkthd_bwd,
        )

        return grad_value, None, None, grad_sampling_loc_attn, None, None

    if T.TYPE_CHECKING:

        @classmethod
        @T.override
        def apply(
            cls,
            value: Tensor,
            shapes: Tensor,
            level_index: Tensor,
            loc_attn: Tensor,
            im2col_step: int,
            points: int,
        ) -> Tensor: ...


class MultiScaleDeformAttnFunction(Function):
    r"""
    Autograd function for Multi-Scale Deformable Attention.
    """

    @staticmethod
    @T.override
    @torch.compiler.disable
    def forward(
        ctx,
        value: Tensor,
        shapes: Tensor,
        level_index: Tensor,
        loc: Tensor,
        attn: Tensor,
        im2col_step: int,
    ):
        ctx.im2col_step = im2col_step
        output = deform_attn_forward(
            value,
            shapes,
            level_index,
            loc,
            attn,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            shapes,
            level_index,
            loc,
            attn,
        )
        return output

    @staticmethod
    @T.override
    @torch.compiler.disable
    @once_differentiable
    def backward(
        ctx, grad_output: Tensor
    ) -> tuple[Tensor, None, None, Tensor, Tensor, None]:
        (
            value,
            spatial_shapes,
            start_index,
            loc,
            attn,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = deform_attn_backward(
            value,
            spatial_shapes,
            start_index,
            loc,
            attn,
            grad_output,
            ctx.im2col_step,
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None

    if T.TYPE_CHECKING:

        @classmethod
        @T.override
        def apply(
            cls,
            value: Tensor,
            shapes: Tensor,
            level_index: Tensor,
            loc: Tensor,
            attn: Tensor,
            im2col_step: int,
        ) -> Tensor: ...


class MultiScaleDeformAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_value: int | None = None,
        dim_output: int | None = None,
        *,
        attention_heads: int,
        levels: int,
        points=4,
        proj: type[nn.Module] = nn.Linear,
        **kwargs,
    ):
        r"""
        Parameters
        ----------
        dim
            Amount of hidden dimension.
        dim_value
            Amount of value dimension, projected to `dim`.
        dim_output
            Amount of output dimensions, projected from `dim`.
        levels
            Amount of feature levels.
        attention_heads
            Amount of attention attention_heads.
        points
            Amount of sampling points per attention head per feature level.
        """
        super().__init__(**kwargs)

        d_per_head = dim // attention_heads
        assert _is_divisible(dim, attention_heads), (dim, attention_heads)
        assert _is_power_of_2(d_per_head), (dim, attention_heads)

        if dim_value is None:
            dim_value = dim
        if dim_output is None:
            dim_output = dim

        self.im2col_step = 128

        self.dim = dim
        self.dim_value = dim_value
        self.dim_output = dim_output
        self.levels = levels
        self.attention_heads = attention_heads
        self.points = points

        self.sampling_offsets = nn.Linear(dim, attention_heads * levels * points * 2)
        self.attention_weights = nn.Linear(dim, attention_heads * levels * points)
        self.proj_value = proj(dim_value, dim)
        self.proj_output = proj(dim, dim_output)

        self.reset_parameters()

    def reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.attention_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.attention_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.attention_heads, 1, 1, 2)
            .repeat(1, self.levels, self.points, 1)
        )
        for i in range(self.points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.reshape(-1).clone())
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.proj_value.weight.data)
        constant_(self.proj_value.bias.data, 0.0)
        xavier_uniform_(self.proj_output.weight.data)
        constant_(self.proj_output.bias.data, 0.0)

    @T.override
    def forward(
        self,
        q: Tensor,
        p: Tensor,
        v: Tensor,
        shapes: Tensor,
        level_index: Tensor,
        padding_mask: Tensor | None = None,
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
        d_batch, d_q, _ = q.shape
        d_batch, d_in, _ = v.shape
        assert (shapes[:, 0] * shapes[:, 1]).sum() == d_in

        v = self.proj_value(v)
        if padding_mask is not None:
            v = v.masked_fill(padding_mask[..., None], float(0))
        v = v.view(
            d_batch, d_in, self.attention_heads, self.dim // self.attention_heads
        )
        attn = self.attention_weights(q).view(
            d_batch, d_q, self.attention_heads, self.levels * self.points
        )
        # attn = softmax(attn, -1)
        loc_off = self.sampling_offsets(q).view(
            d_batch, d_q, self.attention_heads, self.levels, self.points, 2
        )
        if p.shape[-1] == 2:
            loc_norm = torch.stack([shapes[..., 1], shapes[..., 0]], -1)
            loc = (
                p[:, :, None, :, None, :]
                + loc_off / loc_norm[None, None, None, :, None, :]
            )
        elif p.shape[-1] == 4:
            loc = (
                p[:, :, None, :, None, :2]
                + loc_off / self.points * p[:, :, None, :, None, 2:] * 0.5
            )
        else:
            msg = f"Last dim of points must be 2 or 4. Got: {p.shape[-1]}"
            raise ValueError(msg)
        out = self._forward_op(v, shapes, level_index, loc, attn)
        return self.proj_output(out)

    def _forward_op(
        self, v: Tensor, shapes: Tensor, level_index: Tensor, loc: Tensor, attn: Tensor
    ):
        attn = attn.unflatten(-1, (self.levels, self.points))
        with torch.autocast("cuda", enabled=False):
            res = MultiScaleDeformAttnFunction.apply(
                v.float(),
                shapes,
                level_index,
                loc.float(),
                attn.float(),
                self.im2col_step,
            )
        return res.type_as(v)

    if T.TYPE_CHECKING:
        __call__ = forward


class MultiScaleNativeAttn(MultiScaleDeformAttn):
    r"""
    See :class:`MultiScaleDeformAttn` for details.

    Uses native PyTorch implementation.
    """

    def __init__(self, *args, **kwargs):
        """
        See :meth:`MultiScaleDeformAttn.__init__`.
        """
        super().__init__(*args, **kwargs)

        self.im2col_step = None

    @T.override
    def _forward_op(
        self, v: Tensor, shapes: Tensor, level_index: Tensor, loc: Tensor, attn: Tensor
    ) -> Tensor:
        return deform_attn_native(v, shapes, loc, attn)


class MultiScaleFlashAttn(MultiScaleDeformAttn):
    r"""
    See :class:`MultiScaleDeformAttn` for details.

    Uses a FlashAttention-like implementation.
    """

    def __init__(self, *args, **kwargs):
        """
        See :meth:`MultiScaleDeformAttn.__init__`.
        """
        super().__init__(*args, **kwargs)

        self.im2col_step = 64

    @T.override
    def _forward_op(
        self, v: Tensor, shapes: Tensor, level_index: Tensor, loc: Tensor, attn: Tensor
    ) -> Tensor:
        loc = loc.flatten(-3)
        with torch.autocast("cuda", enabled=False):
            loc_attn = torch.cat([loc.half(), attn.half()], dim=-1)
            res = MultiScaleFlashAttnFunction.apply(
                v.half(),
                shapes,
                level_index,
                loc_attn,
                self.im2col_step,
                self.points,
            )
        return res.type_as(v)
        # with torch.autocast("cuda", enabled=False):
        #    loc_attn = torch.cat([loc, attn], dim=-1)
        #    return MultiScaleFlashAttnFunction.apply(
        #        v.half(),
        #        shapes,
        #        level_index,
        #        loc_attn.half(),
        #        self.im2col_step,
        #        self.points,
        #    ).to(dtype=loc.dtype)


torch.fx.wrap("deform_attn_forward")
torch.fx.wrap("deform_attn_backward")
torch.fx.wrap("flash_attn_forward")
torch.fx.wrap("flash_attn_backward")
