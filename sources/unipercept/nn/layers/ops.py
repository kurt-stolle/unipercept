from __future__ import annotations

import torch


# @torch.autocast("cuda", dtype=torch.float16)
def dynamic_conv2d(features: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """
    Perform dynamic convolution on the features.
    Defined in as:
        k ~ (batch num dims)
        f ~ (batch dims h w)
        => dc(k,f) ~ (batch num h w)
    """
    # result = torch.vmap(torch.mm, in_dims=(0, 0))(kernels, features.flatten(2))
    result = torch.bmm(kernels, features.flatten(2))
    result = result.unflatten(2, (features.shape[-2], features.shape[-1]))

    return result


def dynamic_conv2d_16(features: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """
    Uses half precision for the dynamic convolution.
    Currently only supports CUDA tensors. The reason for this is that the
    `torch.bmm` operation is not implemented for half precision on CPU.
    """

    if features.is_cuda:
        return dynamic_conv2d(features.half(), kernels.half()).type_as(features)
    else:
        return dynamic_conv2d(features, kernels)
