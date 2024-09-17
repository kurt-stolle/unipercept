r"""
PyTorch modules and functions for Deformable Convolutional Networks
"""

from __future__ import annotations

import math
import typing as T

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import unipercept.nn.init as weight
from unipercept.nn.activations import ActivationSpec, get_activation
from unipercept.nn.layers import conv
from unipercept.nn.norms import NormSpec, get_norm

from .extension import deform_conv_backward, deform_conv_forward

__all__ = ["DeformConv2d", "DeformConv2dFunction"]

# Precomputed mappings (B, H, W, G, C) -> (d_stride, n_thread, n_block)
type _BHWGCTuple = tuple[int, int, int, int, int]
type _STBTuple = tuple[int, int, int]
type _STBMapping = T.MutableMapping[_BHWGCTuple, _STBTuple]

_FORWARD_LOOKUP: _STBMapping = {
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
_BACKWARD_LOOKUP: _STBMapping = {
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


def _lookup_forward_stride_thread(key: _BHWGCTuple):
    try:
        d_stride, n_thread, n_block = _FORWARD_LOOKUP[key]
        return d_stride, n_thread
    except KeyError:
        pass

    # Heuristic for choosing the forward stride and number of threads
    B, H, W, G, C = key
    d_stride = 8
    ms = factors(B * H * W)
    multiplier = 1
    for m in ms:
        if m <= 64 and (m * G * C // d_stride) <= 512:
            multiplier = m
    n_thread = multiplier * G * C // d_stride
    n_block = (B * H * W + n_thread - 1) // n_thread

    # Write the result into the lookup table
    _FORWARD_LOOKUP[key] = (d_stride, n_thread, n_block)

    return d_stride, n_thread


def _lookup_backward_stride_thread(key: _BHWGCTuple):
    if key in _BACKWARD_LOOKUP:
        d_stride, n_thread, n_block = _BACKWARD_LOOKUP[key]
        return d_stride, n_thread

    # Heuristic for choosing the backward stride and number of threads
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
    n_block = (B * H * W + n_thread - 1) // n_thread

    # Write the result into the lookup table
    _BACKWARD_LOOKUP[key] = (d_stride, n_thread, n_block)

    return d_stride, n_thread


class DeformConv2dFunction(Function):
    @staticmethod
    @T.override
    @torch.compiler.disable
    @custom_fwd(device_type="cuda")
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
        group_dims,
        offset_scale,
        im2col_step,
        remove_center,
        softmax,
    ):
        key = (*input.shape[:3], group, group_dims)
        fwd_stride, fwd_block_thread = _lookup_forward_stride_thread(key)
        bck_stride, bck_block_thread = _lookup_backward_stride_thread(key)

        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride_h = stride_h
        ctx.stride_w = stride_w
        ctx.pad_h = pad_h
        ctx.pad_w = pad_w
        ctx.dilation_h = dilation_h
        ctx.dilation_w = dilation_w
        ctx.group = group
        ctx.group_dims = group_dims
        ctx.offset_scale = offset_scale
        ctx.im2col_step = im2col_step
        ctx.remove_center = remove_center
        ctx.backward_d_stride = bck_stride
        ctx.backward_block_thread = bck_block_thread
        ctx.softmax = softmax

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
            group_dims,
            offset_scale,
            ctx.im2col_step,
            remove_center,
            fwd_stride,
            fwd_block_thread,
            softmax,
        )
        ctx.save_for_backward(input, offset_mask)
        return out

    @staticmethod
    @T.override
    @torch.compiler.disable
    @once_differentiable
    @custom_bwd(device_type="cuda")
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
            ctx.group_dims,
            ctx.offset_scale,
            ctx.im2col_step,
            grad_output.contiguous(),
            ctx.remove_center,
            ctx.backward_d_stride,
            ctx.backward_block_thread,
            ctx.softmax,
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
            None,
        )

    if T.TYPE_CHECKING:

        @classmethod
        @T.override
        def apply(cls, *_: T.Any) -> Tensor: ...


class CenterScale(conv.utils.NormActivationSupport, nn.Module):
    r"""
    Simple wrapper around a linear layer for the purpose of having weight and bias
    parameters that are not named "weight" and "bias" to prevent these parameters
    from being picked up by the optimizer for applying weight decay or other
    transformations.

    The output is passed through a sigmoid function to ensure that the scale is
    between 0 and 1.
    """

    def __init__(self, dims: int, group: int, **kwargs):
        super().__init__(**kwargs)

        self.scale_weight = nn.Parameter(torch.zeros((group, dims), dtype=torch.float))
        self.scale_bias = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float)
            .view((1,))
            .repeat(
                group,
            )
            .clone()
        )

    @T.override
    def forward(self, query: Tensor) -> Tensor:
        return F.linear(
            query,
            weight=self.scale_weight,
            bias=self.scale_bias,
        ).sigmoid()


class DeformConv2d(nn.Module):
    def __init__(
        self,
        dims,
        kernel_size: int = 3,
        *,
        stride: int = 1,
        padding: int | T.Literal["same"] = 1,
        padding_mode: T.Literal["zeros"] = "zeros",
        dilation: int = 1,
        groups: int = 4,
        offset_scale: float = 1.0,
        center_feature_scale: bool = False,
        remove_center: bool = False,
        project: type[nn.Module] | None = None,
        norm: NormSpec = None,
        activation: ActivationSpec = None,
        softmax: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        dims : int
            Number of input dims
        kernel_size : int
            Size of the convolving kernel
        stride : int
            Stride of the convolution
        padding : int
            Padding added to both sides of the input
        padding_mode : str
            Padding mode for the convolutions, currently only "zeros" is supported.
        dilation : int
            Spacing between kernel elements
        groups : int
            Number of blocked connections from input dims to output dims
        offset_scale : float
            Scale of the offset
        center_feature_scale : bool
            Whether to use center feature scale
        remove_center : bool
            Whether to remove the center of the kernel
        bias : bool
            Whether to use bias in the output projection
        project : nn.Module
            Projection layer. Defaults to ``nn.Linear``.
        softmax : bool
            Whether to use softmax in the deformable convolution, defaults to False.
        """
        super().__init__(**kwargs)
        if dims % groups != 0:
            raise ValueError(
                f"dims must be divisible by group, but got {dims} and {groups}"
            )
        _d_per_group = dims // groups
        assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.dims = dims
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        if padding == "same":
            padding = kernel_size // 2
        self.padding = padding
        if padding_mode != "zeros":
            msg = f"padding_mode must be 'zeros', but got {padding_mode}"
            raise ValueError(msg)
        self.padding_mode = padding_mode
        self.groups = groups
        self.group_dims = dims // groups
        self.offset_scale = offset_scale
        self.center_feature_scale = center_feature_scale
        self.remove_center = int(remove_center)
        self.softmax = softmax

        self.K = groups * (kernel_size * kernel_size - self.remove_center)
        self.offset_depthwise = nn.Conv2d(
            dims,
            dims,
            self.kernel_size,
            stride=1,
            padding=self.padding,
            padding_mode="zeros",  # "circular",
            groups=dims,
        )
        self.offset_pointwise = nn.Linear(dims, int(math.ceil((self.K * 3) / 8) * 8))

        if project is not None:
            self.proj_input = project(dims, dims)
            self.proj_output = project(dims, dims)
        else:
            self.register_module("proj_input", None)
            self.register_module("proj_output", None)

        self.norm = get_norm(norm, self.dims)
        self.activation = get_activation(activation)

        self.reset_parameters()

        if center_feature_scale:
            self.center_scale = CenterScale(dims, groups)
        else:
            self.register_module("center_scale", None)

    def reset_parameters(self):
        weight.init_module_(self.offset_pointwise, weight.InitMode.ZEROS)
        weight.init_module_(self.offset_depthwise, weight.InitMode.C2_XAVIER)
        if self.proj_input is not None:
            weight.init_module_(self.proj_input, weight.InitMode.C2_XAVIER)
        if self.proj_output is not None:
            weight.init_module_(self.proj_output, weight.InitMode.C2_XAVIER)

    def _forward_deform(self, out: Tensor, offset_mask: Tensor) -> Tensor:
        return DeformConv2dFunction.apply(
            out,
            offset_mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            self.dilation,
            self.dilation,
            self.groups,
            self.group_dims,
            self.offset_scale,
            256,
            self.remove_center,
            self.softmax,
        )

    @T.override
    def forward(self, input: Tensor, shape: torch.Size | None = None) -> Tensor:
        """
        Parameters
        ----------
        input : Tensor[N, L, C] or Tensor[N, C, H, W]
            The input tensor. It is recommended to use Tensor[N, L, C] for better performance
            wherever this is compatible within the larger architecture context.
        shape : tuple[H,W]
            Shape of the input tensor if input is in the form of Tensor[N, L, C].
            If None, the shape is inferred from the input tensor by assuming it is square.
            When a shape is passed and the input is in the form of Tensor[N, C, H, W],
            the shape is checked against the input shape through an assertion.

        Returns
        -------
        Tensor[N, L, C]
            Result of the deformable convolution
        """

        # Input parsing and transformation
        ndim_input = input.ndim
        if ndim_input == 4:
            assert shape is None
            shape = input.shape[-2:]
            input = input.flatten(2).permute(0, 2, 1).contiguous()
        N, L, C = input.shape
        if shape is not None:
            H, W = shape
        else:
            H, W = int(L**0.5), int(L**0.5)
        out = input

        # Input projection
        if self.proj_input is not None:
            out = self.proj_input(out)

        # Offset mask
        out = out.reshape(N, H, W, -1)
        offset_mask_input = self.offset_depthwise(
            input.view(N, H, W, C).permute(0, 3, 1, 2)
        )
        offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        offset_mask = self.offset_pointwise(offset_mask_input).reshape(N, H, W, -1)

        # Deformable convolution
        out_ante = out
        out = self._forward_deform(out, offset_mask)

        # Center feature scale
        if self.center_scale is not None:
            center_feature_scale = self.center_scale(out)
            center_feature_scale = (
                center_feature_scale[..., None]
                .repeat(1, 1, 1, 1, self.dims // self.groups)
                .flatten(-2)
            )
            out = out * (1 - center_feature_scale) + out_ante * center_feature_scale
        out = out.view(N, L, -1)

        # Output projection
        if self.proj_output is not None:
            out = self.proj_output(out)

        if ndim_input == 4:
            out = out.permute(0, 2, 1).unflatten(2, shape).contiguous()

        if self.norm is not None:
            out = self.norm(out)
        if self.activation is not None:
            out = self.activation(out)

        return out

    if T.TYPE_CHECKING:
        __call__ = forward
