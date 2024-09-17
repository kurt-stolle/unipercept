#pragma once
#include <torch/extension.h>

at::Tensor flash_attn_cuda_forward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
    const int im2col_step, const int K, const int d_stride, const int block_thread);

std::vector<at::Tensor>
flash_attn_cuda_backward(
    const at::Tensor &value, const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index, const at::Tensor &sampling_loc_attn,
    const at::Tensor &grad_output, const int im2col_step, const int K,
    const int d_stride, const int block_thread);
