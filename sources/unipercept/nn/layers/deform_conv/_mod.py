r"""
PyTorch modules and functions for Deformable Convolutional Networks
"""

from __future__ import annotations

import math
import typing as T

import torch
import torch.nn.functional as F
import typing_extensions as TX
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.init import constant_, xavier_uniform_

from .extension import deform_conv_backward, deform_conv_forward

__all__ = ["DeformConv2d", "DeformConv2dFunction"]

# Precomputed mappings (B, H, W, G, C) -> (d_stride, n_thread, n_block)
_BHWGCTuple: T.TypeAlias = T.Tuple[int, int, int, int, int]
_STBTuple: T.TypeAlias = T.Tuple[int, int, int]
_STBMapping: T.TypeAlias = T.MutableMapping[_BHWGCTuple, _STBTuple]

TABLE: _STBMapping = {
    (64, 56, 56, 4, 16): (8, 448, 56),
    (64, 28, 28, 4, 16): (8, 448, 56),
    (64, 14, 14, 4, 16): (8, 32, 4),
    (64, 7, 7, 4, 16): (8, 56, 7),
    (1, 200, 320, 4, 16): (8, 32, 4),
    (1, 100, 160, 4, 16): (8, 32, 4),
    (1, 50, 80, 4, 16): (4, 512, 32),
    (1, 25, 40, 4, 16): (4, 320, 20),
    (1, 64, 64, 4, 16): (8, 512, 64),
    (64, 56, 56, 5, 16): (8, 490, 49),
    (64, 28, 28, 5, 16): (8, 490, 49),
    (64, 14, 14, 5, 16): (8, 280, 28),
    (64, 7, 7, 5, 16): (4, 140, 7),
    (1, 200, 320, 5, 16): (4, 400, 20),
    (1, 100, 160, 5, 16): (4, 400, 20),
    (1, 50, 80, 5, 16): (8, 500, 50),
    (1, 25, 40, 5, 16): (8, 20, 2),
    (1, 64, 64, 5, 16): (8, 320, 32),
    (64, 56, 56, 6, 16): (8, 768, 64),
    (64, 28, 28, 6, 16): (8, 672, 56),
    (64, 14, 14, 6, 16): (8, 336, 28),
    (64, 7, 7, 6, 16): (8, 84, 7),
    (1, 200, 320, 6, 16): (4, 600, 25),
    (1, 100, 160, 6, 16): (4, 600, 25),
    (1, 50, 80, 6, 16): (8, 600, 50),
    (1, 25, 40, 6, 16): (2, 240, 5),
    (1, 64, 64, 6, 16): (8, 384, 32),
    (64, 56, 56, 7, 16): (8, 896, 64),
    (64, 28, 28, 7, 16): (8, 686, 49),
    (64, 14, 14, 7, 16): (8, 392, 28),
    (64, 7, 7, 7, 16): (8, 686, 49),
    (1, 200, 320, 7, 16): (8, 700, 50),
    (1, 100, 160, 7, 16): (8, 700, 50),
    (1, 50, 80, 7, 16): (8, 700, 50),
    (1, 25, 40, 7, 16): (8, 70, 5),
    (1, 64, 64, 7, 16): (8, 448, 32),
    (64, 56, 56, 8, 16): (8, 448, 28),
    (64, 28, 28, 8, 16): (8, 448, 28),
    (64, 14, 14, 8, 16): (8, 448, 28),
    (64, 7, 7, 8, 16): (8, 784, 49),
    (1, 200, 320, 8, 16): (8, 800, 50),
    (1, 100, 160, 8, 16): (4, 640, 20),
    (1, 50, 80, 8, 16): (8, 800, 50),
    (1, 25, 40, 8, 16): (4, 64, 2),
    (1, 64, 64, 8, 16): (8, 256, 16),
    (64, 56, 56, 4, 32): (8, 448, 28),
    (64, 28, 28, 4, 32): (8, 448, 28),
    (64, 14, 14, 4, 32): (8, 448, 28),
    (64, 7, 7, 4, 32): (8, 112, 7),
    (1, 200, 320, 4, 32): (8, 512, 32),
    (1, 100, 160, 4, 32): (8, 800, 50),
    (1, 50, 80, 4, 32): (8, 800, 50),
    (1, 25, 40, 4, 32): (4, 128, 4),
    (1, 64, 64, 4, 32): (8, 128, 8),
    (64, 56, 56, 5, 32): (8, 560, 28),
    (64, 28, 28, 5, 32): (8, 560, 28),
    (64, 14, 14, 5, 32): (8, 560, 28),
    (64, 7, 7, 5, 32): (8, 980, 49),
    (1, 200, 320, 5, 32): (8, 500, 25),
    (1, 100, 160, 5, 32): (8, 800, 40),
    (1, 50, 80, 5, 32): (8, 1000, 50),
    (1, 25, 40, 5, 32): (4, 200, 5),
    (1, 64, 64, 5, 32): (8, 640, 32),
    (64, 56, 56, 6, 32): (8, 336, 14),
    (64, 28, 28, 6, 32): (8, 336, 14),
    (64, 14, 14, 6, 32): (8, 336, 14),
    (64, 7, 7, 6, 32): (16, 588, 49),
    (1, 200, 320, 6, 32): (8, 480, 20),
    (1, 100, 160, 6, 32): (8, 480, 20),
    (1, 50, 80, 6, 32): (16, 600, 50),
    (1, 25, 40, 6, 32): (8, 96, 4),
    (1, 64, 64, 6, 32): (8, 768, 32),
    (64, 56, 56, 7, 32): (8, 448, 16),
    (64, 28, 28, 7, 32): (8, 448, 16),
    (64, 14, 14, 7, 32): (8, 196, 7),
    (64, 7, 7, 7, 32): (8, 28, 1),
    (1, 200, 320, 7, 32): (8, 448, 16),
    (1, 100, 160, 7, 32): (8, 448, 16),
    (1, 50, 80, 7, 32): (8, 700, 25),
    (1, 25, 40, 7, 32): (8, 56, 2),
    (1, 64, 64, 7, 32): (8, 896, 32),
    (64, 56, 56, 8, 32): (8, 448, 14),
    (64, 28, 28, 8, 32): (8, 448, 14),
    (64, 14, 14, 8, 32): (8, 448, 14),
    (64, 7, 7, 8, 32): (8, 32, 1),
    (1, 200, 320, 8, 32): (8, 512, 16),
    (1, 100, 160, 8, 32): (8, 800, 25),
    (1, 50, 80, 8, 32): (8, 800, 25),
    (1, 25, 40, 8, 32): (4, 512, 8),
    (1, 64, 64, 8, 32): (8, 32, 1),
    (64, 56, 56, 4, 64): (8, 448, 14),
    (64, 28, 28, 4, 64): (8, 448, 14),
    (64, 14, 14, 4, 64): (8, 448, 14),
    (64, 7, 7, 4, 64): (8, 32, 1),
    (1, 200, 320, 4, 64): (8, 512, 16),
    (1, 100, 160, 4, 64): (8, 512, 16),
    (1, 50, 80, 4, 64): (8, 800, 25),
    (1, 25, 40, 4, 64): (8, 640, 20),
    (1, 64, 64, 4, 64): (8, 512, 16),
    (64, 56, 56, 5, 64): (8, 560, 14),
    (64, 28, 28, 5, 64): (8, 560, 14),
    (64, 14, 14, 5, 64): (8, 560, 14),
    (64, 7, 7, 5, 64): (8, 280, 7),
    (1, 200, 320, 5, 64): (8, 800, 20),
    (1, 100, 160, 5, 64): (8, 800, 20),
    (1, 50, 80, 5, 64): (8, 1000, 25),
    (1, 25, 40, 5, 64): (8, 80, 2),
    (1, 64, 64, 5, 64): (8, 320, 8),
    (64, 56, 56, 6, 64): (8, 768, 16),
    (64, 28, 28, 6, 64): (8, 768, 16),
    (64, 14, 14, 6, 64): (8, 336, 7),
    (64, 7, 7, 6, 64): (8, 336, 7),
    (1, 200, 320, 6, 64): (8, 768, 16),
    (1, 100, 160, 6, 64): (8, 480, 10),
    (1, 50, 80, 6, 64): (16, 240, 10),
    (1, 25, 40, 6, 64): (8, 240, 5),
    (1, 64, 64, 6, 64): (8, 768, 16),
    (64, 56, 56, 7, 64): (8, 896, 16),
    (64, 28, 28, 7, 64): (8, 448, 8),
    (64, 14, 14, 7, 64): (8, 392, 7),
    (64, 7, 7, 7, 64): (8, 56, 1),
    (1, 200, 320, 7, 64): (8, 896, 16),
    (1, 100, 160, 7, 64): (8, 448, 8),
    (1, 50, 80, 7, 64): (8, 448, 8),
    (1, 25, 40, 7, 64): (8, 448, 8),
    (1, 64, 64, 7, 64): (8, 448, 8),
    (64, 56, 56, 8, 64): (8, 896, 14),
    (64, 28, 28, 8, 64): (8, 896, 14),
    (64, 14, 14, 8, 64): (8, 448, 7),
    (64, 7, 7, 8, 64): (8, 64, 1),
    (1, 200, 320, 8, 64): (8, 512, 8),
    (1, 100, 160, 8, 64): (8, 512, 8),
    (1, 50, 80, 8, 64): (8, 512, 8),
    (1, 25, 40, 8, 64): (8, 512, 8),
    (1, 64, 64, 8, 64): (8, 512, 8),
}
BWDTABLE: _STBMapping = {
    (64, 56, 56, 4, 16): (1, 256, 4),
    (64, 56, 56, 5, 16): (1, 320, 4),
    (64, 56, 56, 6, 16): (1, 192, 2),
    (64, 56, 56, 7, 16): (1, 224, 2),
    (64, 56, 56, 8, 16): (1, 256, 2),
    (64, 56, 56, 4, 32): (1, 256, 2),
    (64, 56, 56, 5, 32): (1, 160, 1),
    (64, 56, 56, 6, 32): (1, 192, 1),
    (64, 56, 56, 7, 32): (1, 224, 1),
    (64, 56, 56, 8, 32): (1, 256, 1),
    (64, 56, 56, 4, 64): (2, 512, 4),
    (64, 56, 56, 5, 64): (2, 640, 4),
    (64, 56, 56, 6, 64): (2, 384, 2),
    (64, 56, 56, 7, 64): (2, 224, 1),
    (64, 56, 56, 8, 64): (2, 1024, 4),
    (64, 28, 28, 4, 16): (1, 128, 2),
    (64, 28, 28, 5, 16): (1, 320, 4),
    (64, 28, 28, 6, 16): (1, 96, 1),
    (64, 28, 28, 7, 16): (1, 224, 2),
    (64, 28, 28, 8, 16): (1, 128, 1),
    (64, 28, 28, 4, 32): (1, 128, 1),
    (64, 28, 28, 5, 32): (1, 320, 2),
    (64, 28, 28, 6, 32): (1, 192, 1),
    (64, 28, 28, 7, 32): (1, 224, 1),
    (64, 28, 28, 8, 32): (1, 256, 1),
    (64, 28, 28, 4, 64): (2, 512, 4),
    (64, 28, 28, 5, 64): (2, 640, 4),
    (64, 28, 28, 6, 64): (2, 384, 2),
    (64, 28, 28, 7, 64): (2, 224, 1),
    (64, 28, 28, 8, 64): (2, 512, 2),
    (64, 14, 14, 4, 16): (1, 128, 2),
    (64, 14, 14, 5, 16): (1, 320, 4),
    (64, 14, 14, 6, 16): (1, 192, 2),
    (64, 14, 14, 7, 16): (1, 224, 2),
    (64, 14, 14, 8, 16): (1, 128, 1),
    (64, 14, 14, 4, 32): (1, 256, 2),
    (64, 14, 14, 5, 32): (1, 160, 1),
    (64, 14, 14, 6, 32): (1, 192, 1),
    (64, 14, 14, 7, 32): (1, 224, 1),
    (64, 14, 14, 8, 32): (1, 256, 1),
    (64, 14, 14, 4, 64): (2, 128, 1),
    (64, 14, 14, 5, 64): (2, 160, 1),
    (64, 14, 14, 6, 64): (2, 384, 2),
    (64, 14, 14, 7, 64): (2, 224, 1),
    (64, 14, 14, 8, 64): (2, 256, 1),
    (64, 7, 7, 4, 16): (4, 784, 49),
    (64, 7, 7, 5, 16): (2, 280, 7),
    (64, 7, 7, 6, 16): (2, 48, 1),
    (64, 7, 7, 7, 16): (2, 392, 7),
    (64, 7, 7, 8, 16): (1, 128, 1),
    (64, 7, 7, 4, 32): (1, 128, 1),
    (64, 7, 7, 5, 32): (1, 160, 1),
    (64, 7, 7, 6, 32): (2, 96, 1),
    (64, 7, 7, 7, 32): (2, 112, 1),
    (64, 7, 7, 8, 32): (2, 128, 1),
    (64, 7, 7, 4, 64): (2, 896, 7),
    (64, 7, 7, 5, 64): (2, 160, 1),
    (64, 7, 7, 6, 64): (2, 192, 1),
    (64, 7, 7, 7, 64): (2, 224, 1),
    (64, 7, 7, 8, 64): (2, 256, 1),
    (1, 200, 320, 4, 16): (1, 320, 5),
    (1, 200, 320, 5, 16): (1, 320, 4),
    (1, 200, 320, 6, 16): (1, 96, 1),
    (1, 200, 320, 7, 16): (1, 224, 2),
    (1, 200, 320, 8, 16): (1, 640, 5),
    (1, 200, 320, 4, 32): (1, 128, 1),
    (1, 200, 320, 5, 32): (1, 320, 2),
    (1, 200, 320, 6, 32): (1, 384, 2),
    (1, 200, 320, 7, 32): (1, 224, 1),
    (1, 200, 320, 8, 32): (1, 256, 1),
    (1, 200, 320, 4, 64): (2, 640, 5),
    (1, 200, 320, 5, 64): (2, 800, 5),
    (1, 200, 320, 6, 64): (2, 768, 4),
    (1, 200, 320, 7, 64): (2, 448, 2),
    (1, 200, 320, 8, 64): (2, 1024, 4),
    (1, 100, 160, 4, 16): (1, 320, 5),
    (1, 100, 160, 5, 16): (1, 640, 8),
    (1, 100, 160, 6, 16): (1, 96, 1),
    (1, 100, 160, 7, 16): (1, 224, 2),
    (1, 100, 160, 8, 16): (1, 640, 5),
    (1, 100, 160, 4, 32): (1, 256, 2),
    (1, 100, 160, 5, 32): (1, 160, 1),
    (1, 100, 160, 6, 32): (1, 384, 2),
    (1, 100, 160, 7, 32): (1, 224, 1),
    (1, 100, 160, 8, 32): (1, 512, 2),
    (1, 100, 160, 4, 64): (2, 128, 1),
    (1, 100, 160, 5, 64): (2, 160, 1),
    (1, 100, 160, 6, 64): (2, 384, 2),
    (1, 100, 160, 7, 64): (2, 448, 2),
    (1, 100, 160, 8, 64): (2, 512, 2),
    (1, 50, 80, 4, 16): (1, 320, 5),
    (1, 50, 80, 5, 16): (1, 320, 4),
    (1, 50, 80, 6, 16): (1, 96, 1),
    (1, 50, 80, 7, 16): (1, 112, 1),
    (1, 50, 80, 8, 16): (1, 512, 4),
    (1, 50, 80, 4, 32): (1, 128, 1),
    (1, 50, 80, 5, 32): (1, 320, 2),
    (1, 50, 80, 6, 32): (1, 384, 2),
    (1, 50, 80, 7, 32): (1, 224, 1),
    (1, 50, 80, 8, 32): (1, 256, 1),
    (1, 50, 80, 4, 64): (2, 256, 2),
    (1, 50, 80, 5, 64): (2, 640, 4),
    (1, 50, 80, 6, 64): (2, 768, 4),
    (1, 50, 80, 7, 64): (2, 448, 2),
    (1, 50, 80, 8, 64): (2, 1024, 4),
    (1, 25, 40, 4, 16): (1, 320, 5),
    (1, 25, 40, 5, 16): (2, 400, 10),
    (1, 25, 40, 6, 16): (1, 192, 2),
    (1, 25, 40, 7, 16): (4, 224, 8),
    (1, 25, 40, 8, 16): (4, 160, 5),
    (1, 25, 40, 4, 32): (2, 128, 2),
    (1, 25, 40, 5, 32): (1, 320, 2),
    (1, 25, 40, 6, 32): (2, 96, 1),
    (1, 25, 40, 7, 32): (2, 112, 1),
    (1, 25, 40, 8, 32): (2, 640, 5),
    (1, 25, 40, 4, 64): (2, 128, 1),
    (1, 25, 40, 5, 64): (2, 160, 1),
    (1, 25, 40, 6, 64): (2, 192, 1),
    (1, 25, 40, 7, 64): (2, 896, 4),
    (1, 25, 40, 8, 64): (2, 512, 2),
    (1, 64, 64, 4, 16): (1, 256, 4),
    (1, 64, 64, 5, 16): (2, 40, 1),
    (1, 64, 64, 6, 16): (1, 192, 2),
    (1, 64, 64, 7, 16): (1, 224, 2),
    (1, 64, 64, 8, 16): (1, 512, 4),
    (1, 64, 64, 4, 32): (2, 64, 1),
    (1, 64, 64, 5, 32): (1, 320, 2),
    (1, 64, 64, 6, 32): (1, 192, 1),
    (1, 64, 64, 7, 32): (1, 224, 1),
    (1, 64, 64, 8, 32): (1, 256, 1),
    (1, 64, 64, 4, 64): (2, 512, 4),
    (1, 64, 64, 5, 64): (2, 640, 4),
    (1, 64, 64, 6, 64): (2, 192, 1),
    (1, 64, 64, 7, 64): (2, 224, 1),
    (1, 64, 64, 8, 64): (2, 256, 1),
}


def factors(N):
    res = []
    for i in range(1, N + 1):
        if N % i == 0:
            res.append(i)
    return res


def findspec(*key: int):
    try:
        S, T, B = TABLE[key]
        return S, T
    except KeyError:
        pass

    B, H, W, G, C = key
    d_stride = 8
    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    TABLE[(B, H, W, G, C)] = (d_stride, n_thread)
    return d_stride, n_thread


def find_spec_bwd(*key):
    if key in BWDTABLE:
        return BWDTABLE[key][0], BWDTABLE[key][1]
    B, H, W, G, C = key
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


class DeformConv2dFunction(Function):
    @staticmethod
    @TX.override
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
        fwd_stride, fwd_block_thread = findspec(
            input.shape[0], input.shape[1], input.shape[2], group, group_channels
        )
        bck_stride, bck_block_thread = find_spec_bwd(
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
        ctx.backward_d_stride = bck_stride
        ctx.backward_block_thread = bck_block_thread

        out = deform_conv_forward(
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
            fwd_stride,
            fwd_block_thread,
            False,
        )
        ctx.save_for_backward(input, offset_mask)
        return out

    @staticmethod
    @TX.override
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output: Tensor):
        input, offset_mask = ctx.saved_tensors
        grad_input, grad_offset_mask = deform_conv_backward(
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
        )
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

    if T.TYPE_CHECKING:

        @classmethod
        @TX.override
        def apply(cls, *_: T.Any) -> Tensor: ...


class CenterFeatureScaleModule(nn.Module):
    @TX.override
    def forward(self, query: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        center_feature_scale = F.linear(
            query,
            weight=weight,
            bias=bias,
        ).sigmoid()
        return center_feature_scale


class DeformConv2d(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        *,
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
        Parameters
        ----------
        channels : int
            Number of input channels
        kernel_size : int
            Size of the convolving kernel
        stride : int
            Stride of the convolution
        pad : int
            Zero-padding added to both sides of the input
        dilation : int
            Spacing between kernel elements
        group : int
            Number of blocked connections from input channels to output channels
        offset_scale : float
            Scale of the offset
        dw_kernel_size : int
            Size of the depthwise kernel
        center_feature_scale : bool
            Whether to use center feature scale
        remove_center : bool
            Whether to remove the center of the kernel
        output_bias : bool
            Whether to use bias in the output projection
        without_pointwise : bool
            Whether to use pointwise projection
        """
        super().__init__(**kwargs)
        if channels % group != 0:
            raise ValueError(
                f"channels must be divisible by group, but got {channels} and {group}"
            )
        _d_per_group = channels // group
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
        else:
            self.register_module("value_proj", None)
            self.register_module("output_proj", None)

        self._reset_parameters()

        if center_feature_scale:
            self.center_scale_weight = nn.Parameter(
                torch.zeros((group, channels), dtype=torch.float)
            )
            self.center_scale_bias = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float)
                .view((1,))
                .repeat(
                    group,
                )
            )
            self.center_scale = CenterFeatureScaleModule()
        else:
            self.register_parameter("center_scale_weight", None)
            self.register_parameter("center_scale_bias", None)
            self.register_module("center_scale", None)

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.0)
        constant_(self.offset_mask.bias.data, 0.0)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.0)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.0)

    @TX.override
    def forward(self, input: Tensor, shape: torch.Size | None = None) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor[N, L, C] or Tensor[N, C, H, W]
            The input tensor.
        shape : tuple[H,W]
            Shape of the input tensor if input is in the form of Tensor[N, L, C]

        Returns
        -------
        Tensor[N, L, C]
            Result of the deformable convolution

        """
        N, L, C = input.shape
        if shape is not None:
            H, W = shape
        else:
            H, W = int(L**0.5), int(L**0.5)

        x = input
        if self.value_proj is not None:
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
        x = DeformConv2dFunction.apply(
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

        if self.center_scale is not None:
            center_feature_scale = self.center_scale(
                x,
                self.center_scale_weight,
                self.center_scale_bias,
            )
            center_feature_scale = (
                center_feature_scale[..., None]
                .repeat(1, 1, 1, 1, self.channels // self.group)
                .flatten(-2)
            )
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = x.view(N, L, -1)

        if self.output_proj is not None:
            x = self.output_proj(x)

        return x
