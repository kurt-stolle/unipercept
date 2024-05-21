"""
Deformable Convolutions with Modulation
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.init import constant_, xavier_uniform_

from ._ext import deform_conv_backward, deform_conv_forward

TABLE = {
    "64x56x56x4x16": [8, 448, 56],
    "64x28x28x4x16": [8, 448, 56],
    "64x14x14x4x16": [8, 32, 4],
    "64x7x7x4x16": [8, 56, 7],
    "1x200x320x4x16": [8, 32, 4],
    "1x100x160x4x16": [8, 32, 4],
    "1x50x80x4x16": [4, 512, 32],
    "1x25x40x4x16": [4, 320, 20],
    "1x64x64x4x16": [8, 512, 64],
    "64x56x56x5x16": [8, 490, 49],
    "64x28x28x5x16": [8, 490, 49],
    "64x14x14x5x16": [8, 280, 28],
    "64x7x7x5x16": [4, 140, 7],
    "1x200x320x5x16": [4, 400, 20],
    "1x100x160x5x16": [4, 400, 20],
    "1x50x80x5x16": [8, 500, 50],
    "1x25x40x5x16": [8, 20, 2],
    "1x64x64x5x16": [8, 320, 32],
    "64x56x56x6x16": [8, 768, 64],
    "64x28x28x6x16": [8, 672, 56],
    "64x14x14x6x16": [8, 336, 28],
    "64x7x7x6x16": [8, 84, 7],
    "1x200x320x6x16": [4, 600, 25],
    "1x100x160x6x16": [4, 600, 25],
    "1x50x80x6x16": [8, 600, 50],
    "1x25x40x6x16": [2, 240, 5],
    "1x64x64x6x16": [8, 384, 32],
    "64x56x56x7x16": [8, 896, 64],
    "64x28x28x7x16": [8, 686, 49],
    "64x14x14x7x16": [8, 392, 28],
    "64x7x7x7x16": [8, 686, 49],
    "1x200x320x7x16": [8, 700, 50],
    "1x100x160x7x16": [8, 700, 50],
    "1x50x80x7x16": [8, 700, 50],
    "1x25x40x7x16": [8, 70, 5],
    "1x64x64x7x16": [8, 448, 32],
    "64x56x56x8x16": [8, 448, 28],
    "64x28x28x8x16": [8, 448, 28],
    "64x14x14x8x16": [8, 448, 28],
    "64x7x7x8x16": [8, 784, 49],
    "1x200x320x8x16": [8, 800, 50],
    "1x100x160x8x16": [4, 640, 20],
    "1x50x80x8x16": [8, 800, 50],
    "1x25x40x8x16": [4, 64, 2],
    "1x64x64x8x16": [8, 256, 16],
    "64x56x56x4x32": [8, 448, 28],
    "64x28x28x4x32": [8, 448, 28],
    "64x14x14x4x32": [8, 448, 28],
    "64x7x7x4x32": [8, 112, 7],
    "1x200x320x4x32": [8, 512, 32],
    "1x100x160x4x32": [8, 800, 50],
    "1x50x80x4x32": [8, 800, 50],
    "1x25x40x4x32": [4, 128, 4],
    "1x64x64x4x32": [8, 128, 8],
    "64x56x56x5x32": [8, 560, 28],
    "64x28x28x5x32": [8, 560, 28],
    "64x14x14x5x32": [8, 560, 28],
    "64x7x7x5x32": [8, 980, 49],
    "1x200x320x5x32": [8, 500, 25],
    "1x100x160x5x32": [8, 800, 40],
    "1x50x80x5x32": [8, 1000, 50],
    "1x25x40x5x32": [4, 200, 5],
    "1x64x64x5x32": [8, 640, 32],
    "64x56x56x6x32": [8, 336, 14],
    "64x28x28x6x32": [8, 336, 14],
    "64x14x14x6x32": [8, 336, 14],
    "64x7x7x6x32": [16, 588, 49],
    "1x200x320x6x32": [8, 480, 20],
    "1x100x160x6x32": [8, 480, 20],
    "1x50x80x6x32": [16, 600, 50],
    "1x25x40x6x32": [8, 96, 4],
    "1x64x64x6x32": [8, 768, 32],
    "64x56x56x7x32": [8, 448, 16],
    "64x28x28x7x32": [8, 448, 16],
    "64x14x14x7x32": [8, 196, 7],
    "64x7x7x7x32": [8, 28, 1],
    "1x200x320x7x32": [8, 448, 16],
    "1x100x160x7x32": [8, 448, 16],
    "1x50x80x7x32": [8, 700, 25],
    "1x25x40x7x32": [8, 56, 2],
    "1x64x64x7x32": [8, 896, 32],
    "64x56x56x8x32": [8, 448, 14],
    "64x28x28x8x32": [8, 448, 14],
    "64x14x14x8x32": [8, 448, 14],
    "64x7x7x8x32": [8, 32, 1],
    "1x200x320x8x32": [8, 512, 16],
    "1x100x160x8x32": [8, 800, 25],
    "1x50x80x8x32": [8, 800, 25],
    "1x25x40x8x32": [4, 512, 8],
    "1x64x64x8x32": [8, 32, 1],
    "64x56x56x4x64": [8, 448, 14],
    "64x28x28x4x64": [8, 448, 14],
    "64x14x14x4x64": [8, 448, 14],
    "64x7x7x4x64": [8, 32, 1],
    "1x200x320x4x64": [8, 512, 16],
    "1x100x160x4x64": [8, 512, 16],
    "1x50x80x4x64": [8, 800, 25],
    "1x25x40x4x64": [8, 640, 20],
    "1x64x64x4x64": [8, 512, 16],
    "64x56x56x5x64": [8, 560, 14],
    "64x28x28x5x64": [8, 560, 14],
    "64x14x14x5x64": [8, 560, 14],
    "64x7x7x5x64": [8, 280, 7],
    "1x200x320x5x64": [8, 800, 20],
    "1x100x160x5x64": [8, 800, 20],
    "1x50x80x5x64": [8, 1000, 25],
    "1x25x40x5x64": [8, 80, 2],
    "1x64x64x5x64": [8, 320, 8],
    "64x56x56x6x64": [8, 768, 16],
    "64x28x28x6x64": [8, 768, 16],
    "64x14x14x6x64": [8, 336, 7],
    "64x7x7x6x64": [8, 336, 7],
    "1x200x320x6x64": [8, 768, 16],
    "1x100x160x6x64": [8, 480, 10],
    "1x50x80x6x64": [16, 240, 10],
    "1x25x40x6x64": [8, 240, 5],
    "1x64x64x6x64": [8, 768, 16],
    "64x56x56x7x64": [8, 896, 16],
    "64x28x28x7x64": [8, 448, 8],
    "64x14x14x7x64": [8, 392, 7],
    "64x7x7x7x64": [8, 56, 1],
    "1x200x320x7x64": [8, 896, 16],
    "1x100x160x7x64": [8, 448, 8],
    "1x50x80x7x64": [8, 448, 8],
    "1x25x40x7x64": [8, 448, 8],
    "1x64x64x7x64": [8, 448, 8],
    "64x56x56x8x64": [8, 896, 14],
    "64x28x28x8x64": [8, 896, 14],
    "64x14x14x8x64": [8, 448, 7],
    "64x7x7x8x64": [8, 64, 1],
    "1x200x320x8x64": [8, 512, 8],
    "1x100x160x8x64": [8, 512, 8],
    "1x50x80x8x64": [8, 512, 8],
    "1x25x40x8x64": [8, 512, 8],
    "1x64x64x8x64": [8, 512, 8],
}

BWDTABLE = {
    "64x56x56x4x16": [1, 256, 4],
    "64x56x56x5x16": [1, 320, 4],
    "64x56x56x6x16": [1, 192, 2],
    "64x56x56x7x16": [1, 224, 2],
    "64x56x56x8x16": [1, 256, 2],
    "64x56x56x4x32": [1, 256, 2],
    "64x56x56x5x32": [1, 160, 1],
    "64x56x56x6x32": [1, 192, 1],
    "64x56x56x7x32": [1, 224, 1],
    "64x56x56x8x32": [1, 256, 1],
    "64x56x56x4x64": [2, 512, 4],
    "64x56x56x5x64": [2, 640, 4],
    "64x56x56x6x64": [2, 384, 2],
    "64x56x56x7x64": [2, 224, 1],
    "64x56x56x8x64": [2, 1024, 4],
    "64x28x28x4x16": [1, 128, 2],
    "64x28x28x5x16": [1, 320, 4],
    "64x28x28x6x16": [1, 96, 1],
    "64x28x28x7x16": [1, 224, 2],
    "64x28x28x8x16": [1, 128, 1],
    "64x28x28x4x32": [1, 128, 1],
    "64x28x28x5x32": [1, 320, 2],
    "64x28x28x6x32": [1, 192, 1],
    "64x28x28x7x32": [1, 224, 1],
    "64x28x28x8x32": [1, 256, 1],
    "64x28x28x4x64": [2, 512, 4],
    "64x28x28x5x64": [2, 640, 4],
    "64x28x28x6x64": [2, 384, 2],
    "64x28x28x7x64": [2, 224, 1],
    "64x28x28x8x64": [2, 512, 2],
    "64x14x14x4x16": [1, 128, 2],
    "64x14x14x5x16": [1, 320, 4],
    "64x14x14x6x16": [1, 192, 2],
    "64x14x14x7x16": [1, 224, 2],
    "64x14x14x8x16": [1, 128, 1],
    "64x14x14x4x32": [1, 256, 2],
    "64x14x14x5x32": [1, 160, 1],
    "64x14x14x6x32": [1, 192, 1],
    "64x14x14x7x32": [1, 224, 1],
    "64x14x14x8x32": [1, 256, 1],
    "64x14x14x4x64": [2, 128, 1],
    "64x14x14x5x64": [2, 160, 1],
    "64x14x14x6x64": [2, 384, 2],
    "64x14x14x7x64": [2, 224, 1],
    "64x14x14x8x64": [2, 256, 1],
    "64x7x7x4x16": [4, 784, 49],
    "64x7x7x5x16": [2, 280, 7],
    "64x7x7x6x16": [2, 48, 1],
    "64x7x7x7x16": [2, 392, 7],
    "64x7x7x8x16": [1, 128, 1],
    "64x7x7x4x32": [1, 128, 1],
    "64x7x7x5x32": [1, 160, 1],
    "64x7x7x6x32": [2, 96, 1],
    "64x7x7x7x32": [2, 112, 1],
    "64x7x7x8x32": [2, 128, 1],
    "64x7x7x4x64": [2, 896, 7],
    "64x7x7x5x64": [2, 160, 1],
    "64x7x7x6x64": [2, 192, 1],
    "64x7x7x7x64": [2, 224, 1],
    "64x7x7x8x64": [2, 256, 1],
    "1x200x320x4x16": [1, 320, 5],
    "1x200x320x5x16": [1, 320, 4],
    "1x200x320x6x16": [1, 96, 1],
    "1x200x320x7x16": [1, 224, 2],
    "1x200x320x8x16": [1, 640, 5],
    "1x200x320x4x32": [1, 128, 1],
    "1x200x320x5x32": [1, 320, 2],
    "1x200x320x6x32": [1, 384, 2],
    "1x200x320x7x32": [1, 224, 1],
    "1x200x320x8x32": [1, 256, 1],
    "1x200x320x4x64": [2, 640, 5],
    "1x200x320x5x64": [2, 800, 5],
    "1x200x320x6x64": [2, 768, 4],
    "1x200x320x7x64": [2, 448, 2],
    "1x200x320x8x64": [2, 1024, 4],
    "1x100x160x4x16": [1, 320, 5],
    "1x100x160x5x16": [1, 640, 8],
    "1x100x160x6x16": [1, 96, 1],
    "1x100x160x7x16": [1, 224, 2],
    "1x100x160x8x16": [1, 640, 5],
    "1x100x160x4x32": [1, 256, 2],
    "1x100x160x5x32": [1, 160, 1],
    "1x100x160x6x32": [1, 384, 2],
    "1x100x160x7x32": [1, 224, 1],
    "1x100x160x8x32": [1, 512, 2],
    "1x100x160x4x64": [2, 128, 1],
    "1x100x160x5x64": [2, 160, 1],
    "1x100x160x6x64": [2, 384, 2],
    "1x100x160x7x64": [2, 448, 2],
    "1x100x160x8x64": [2, 512, 2],
    "1x50x80x4x16": [1, 320, 5],
    "1x50x80x5x16": [1, 320, 4],
    "1x50x80x6x16": [1, 96, 1],
    "1x50x80x7x16": [1, 112, 1],
    "1x50x80x8x16": [1, 512, 4],
    "1x50x80x4x32": [1, 128, 1],
    "1x50x80x5x32": [1, 320, 2],
    "1x50x80x6x32": [1, 384, 2],
    "1x50x80x7x32": [1, 224, 1],
    "1x50x80x8x32": [1, 256, 1],
    "1x50x80x4x64": [2, 256, 2],
    "1x50x80x5x64": [2, 640, 4],
    "1x50x80x6x64": [2, 768, 4],
    "1x50x80x7x64": [2, 448, 2],
    "1x50x80x8x64": [2, 1024, 4],
    "1x25x40x4x16": [1, 320, 5],
    "1x25x40x5x16": [2, 400, 10],
    "1x25x40x6x16": [1, 192, 2],
    "1x25x40x7x16": [4, 224, 8],
    "1x25x40x8x16": [4, 160, 5],
    "1x25x40x4x32": [2, 128, 2],
    "1x25x40x5x32": [1, 320, 2],
    "1x25x40x6x32": [2, 96, 1],
    "1x25x40x7x32": [2, 112, 1],
    "1x25x40x8x32": [2, 640, 5],
    "1x25x40x4x64": [2, 128, 1],
    "1x25x40x5x64": [2, 160, 1],
    "1x25x40x6x64": [2, 192, 1],
    "1x25x40x7x64": [2, 896, 4],
    "1x25x40x8x64": [2, 512, 2],
    "1x64x64x4x16": [1, 256, 4],
    "1x64x64x5x16": [2, 40, 1],
    "1x64x64x6x16": [1, 192, 2],
    "1x64x64x7x16": [1, 224, 2],
    "1x64x64x8x16": [1, 512, 4],
    "1x64x64x4x32": [2, 64, 1],
    "1x64x64x5x32": [1, 320, 2],
    "1x64x64x6x32": [1, 192, 1],
    "1x64x64x7x32": [1, 224, 1],
    "1x64x64x8x32": [1, 256, 1],
    "1x64x64x4x64": [2, 512, 4],
    "1x64x64x5x64": [2, 640, 4],
    "1x64x64x6x64": [2, 192, 1],
    "1x64x64x7x64": [2, 224, 1],
    "1x64x64x8x64": [2, 256, 1],
}


def factors(N):
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def findspec(B, H, W, G, C):
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in TABLE:
        return TABLE[key][0], TABLE[key][1]

    d_stride = 8
    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    key = f"{B}x{H}x{W}x{G}x{C}"
    TABLE[key] = (d_stride, n_thread)
    return d_stride, n_thread


def find_spec_bwd(B, H, W, G, C):
    key = f"{B}x{H}x{W}x{G}x{C}"
    if key in BWDTABLE:
        return BWDTABLE[key][0], BWDTABLE[key][1]

    if C >= 64:
        d_stride = 2
    else:
        d_stride = 1

    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 256:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    return d_stride, n_thread


class DeformConvFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        offset_mask,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        group,
        group_channels,
        offset_scale,
        im2col_step,
        remove_center,
    ):
        forward_d_stride, forward_block_thread = findspec(
            input.shape[0], input.shape[1], input.shape[2], group, group_channels
        )
        backward_d_stride, backward_block_thread = find_spec_bwd(
            input.shape[0], input.shape[1], input.shape[2], group, group_channels
        )

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_channels = group_channels
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.backward_d_stride = backward_d_stride
        ctx.backward_block_thread = backward_block_thread

        args = [
            input,
            offset_mask,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_channels,
            offset_scale,
            ctx.im2col_step,
            remove_center,
            forward_d_stride,
            forward_block_thread,
            False,
        ]

        output = deform_conv_forward(*args)
        ctx.save_for_backward(input, offset_mask)

        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        input, offset_mask = ctx.saved_tensors

        args = [
            input,
            offset_mask,
            ctx.kernel_h,
            ctx.kernel_w,
            ctx.stride_h,
            ctx.stride_w,
            ctx.pad_h,
            ctx.pad_w,
            ctx.dilation_h,
            ctx.dilation_w,
            ctx.group,
            ctx.group_channels,
            ctx.offset_scale,
            ctx.im2col_step,
            grad_output.contiguous(),
            ctx.remove_center,
            ctx.backward_d_stride,
            ctx.backward_block_thread,
            False,
        ]

        grad_input, grad_offset_mask = deform_conv_backward(*args)

        return (
            grad_input,
            grad_offset_mask,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CenterFeatureScaleModule(nn.Module):
    def forward(
        self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias
    ):
        center_feature_scale = F.linear(
            query,
            weight=center_feature_scale_proj_weight,
            bias=center_feature_scale_proj_bias,
        ).sigmoid()
        return center_feature_scale


class DeformConv(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        dw_kernel_size=None,
        center_feature_scale=False,
        remove_center=False,
        output_bias=True,
        without_pointwise=False,
        **kwargs,
    ):
        """
        DeformConv Module
        :param channels
        :param kernel_size
        :param stride
        :param pad
        :param dilation
        :param group
        :param offset_scale
        :param act_layer
        :param norm_layer
        """
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f"channels must be divisible by group, but got {channels} and {group}"
            )
        _d_per_group = channels // group

        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.K = group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(
                channels,
                channels,
                dw_kernel_size,
                stride=1,
                padding=(dw_kernel_size - 1) // 2,
                groups=channels,
            )
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3) / 8) * 8))
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float)
            )
            self.center_feature_scale_proj_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float)
                .view((1,))
                .repeat(
                    group,
                )
            )
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.0)
        constant_(self.offset_mask.bias.data, 0.0)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.0)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.0)

    def forward(self, input, shape=None):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, L, C = input.shape
        if shape is not None:
            H, W = shape
        else:
            H, W = int(L**0.5), int(L**0.5)

        x = input
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1)
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(
                input.view(N, H, W, C).permute(0, 3, 1, 2)
            )
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)

        x_proj = x

        x = DeformConvFunction.apply(
            x,
            offset_mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale,
            256,
            self.remove_center,
        )

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x,
                self.center_feature_scale_proj_weight,
                self.center_feature_scale_proj_bias,
            )
            center_feature_scale = (
                center_feature_scale[..., None]
                .repeat(1, 1, 1, 1, self.channels // self.group)
                .flatten(-2)
            )
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale

        x = x.view(N, L, -1)

        if not self.without_pointwise:
            x = self.output_proj(x)
        return x
