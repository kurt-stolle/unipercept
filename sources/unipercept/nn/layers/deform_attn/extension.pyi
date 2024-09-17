from __future__ import annotations

from torch import Tensor

def deform_attn_backward(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_loc: Tensor,
    attn_weight: Tensor,
    grad_output: Tensor,
    im2col_step: int,
    /,
) -> Tensor: ...
def deform_attn_forward(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_loc: Tensor,
    attn_weight: Tensor,
    im2col_step: int,
    /,
) -> Tensor: ...
def flash_attn_backward(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_loc_attn: Tensor,
    grad_output: Tensor,
    im2col_step: int,
    K: int,
    d_stride: int,
    block_thread: int,
) -> Tensor: ...
def flash_attn_forward(
    value: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    sampling_loc_attn: Tensor,
    im2col_step: int,
    K: int,
    d_stride: int,
    block_thread: int,
) -> Tensor: ...
