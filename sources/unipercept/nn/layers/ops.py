from __future__ import annotations

import torch
import torch.fx


# @torch.autocast("cuda", dtype=torch.float16)
def dynamic_conv2d(features: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """
    Perform dynamic convolution on the features.
    Defined in as:
        k ~ (batch num dims)
        f ~ (batch dims h w)
        => dc(k,f) ~ (batch num h w)
    """
    hw = (features.shape[-2], features.shape[-1])

    kernels = kernels.contiguous()

    if kernels.ndim == 3:
        # Batched version
        result = torch.bmm(kernels, features.flatten(2).contiguous())
        result = result.unflatten(2, hw)

    elif kernels.ndim == 2:
        # Unbatched version
        result = torch.mm(kernels, features.flatten(1).contiguous())
        result = result.unflatten(1, hw)
    else:
        msg = f"Unsupported kernel shape: {kernels.shape}"
        raise NotImplementedError(msg)

    return result.contiguous()


torch.fx.wrap("dynamic_conv2d")
