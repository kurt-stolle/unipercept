#pragma once

#include <torch/extension.h>
#include <tuple>
#include <vector>

#ifdef WITH_CUDA
#include "cuda/deform_attn.h"
#include "cuda/flash_attn.h"
#endif

#define ERROR_NO_GPU "Not compiled with GPU support"
#define ERROR_NO_CPU "Not implemented on the CPU"

/* --------------------
 * Deformable Attention
 * -------------------- */

at::Tensor deform_attn_forward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int im2col_step)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_attn_cuda_forward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step);
#else
        AT_ERROR(ERROR_NO_GPU);
#endif
    }
    AT_ERROR(ERROR_NO_CPU);
}

std::vector<at::Tensor>
deform_attn_backward(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const at::Tensor &grad_output,
    const int im2col_step)
{
    if (value.type().is_cuda())
    {
#ifdef WITH_CUDA
        return deform_attn_cuda_backward(
            value, spatial_shapes, level_start_index, sampling_loc, attn_weight, grad_output, im2col_step);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    AT_ERROR(ERROR_NO_CPU);
}

/* --------------------
 * Flash Attention
 * -------------------- */

at::Tensor flash_attn_forward(const at::Tensor &value,
                                        const at::Tensor &spatial_shapes,
                                        const at::Tensor &level_start_index,
                                        const at::Tensor &sampling_loc_attn,
                                        const int im2col_step, const int K,
                                        const int d_stride, const int block_thread) {
  if (value.device().is_cuda()) {
#ifdef WITH_CUDA
    return flash_attn_cuda_forward(value, spatial_shapes,
                                             level_start_index,
                                             sampling_loc_attn, im2col_step, 
                                             K, d_stride, block_thread);
#else
    AT_ERROR(ERROR_NO_GPU);
#endif
  }
  AT_ERROR(ERROR_NO_CPU);
}

std::vector<at::Tensor> flash_attn_backward(const at::Tensor &value, 
                              const at::Tensor &spatial_shapes,
                              const at::Tensor &level_start_index, 
                              const at::Tensor &sampling_loc_attn,
                              const at::Tensor &grad_output, 
                              const int im2col_step, 
                              const int K, 
                              const int d_stride, const int block_thread){
  if (value.device().is_cuda()) {
#ifdef WITH_CUDA
    return flash_attn_cuda_backward(value, 
                                              spatial_shapes,
                                              level_start_index,
                                              sampling_loc_attn, 
                                              grad_output,
                                              im2col_step, 
                                              K, d_stride, 
                                              block_thread);
#else
    AT_ERROR(ERROR_NO_GPU);
#endif
  }
  AT_ERROR(ERROR_NO_CPU);
}

