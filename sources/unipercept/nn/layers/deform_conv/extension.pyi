"""
Interface for the extension module.

See Also
--------

- ``extension.h`` : reference header file.

"""

from __future__ import annotations

from torch import Tensor

def deform_conv_backward(
    value: Tensor,
    p_offset: Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    group: int,
    group_channels: int,
    offset_scale: float,
    im2col_step: int,
    grad_output: Tensor,
    remove_center: int,
    d_stride: int,
    block_thread: int,
    softmax: bool,
    /,
) -> Tensor: ...
def deform_conv_forward(
    value: Tensor,
    p_offset: Tensor,
    kernel_h: int,
    kernel_w: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dilation_h: int,
    dilation_w: int,
    group: int,
    group_channels: int,
    offset_scale: float,
    im2col_step: int,
    remove_center: int,
    d_stride: int,
    block_thread: int,
    softmax: bool,
    /,
) -> Tensor: ...
