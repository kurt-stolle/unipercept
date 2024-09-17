from __future__ import annotations

import torch
from torch import Tensor

torch.library.custom_op("unipercept::deform_conv", mutates_args=())


def deform_conv(x: Tensor) -> Tensor:
    return x.sin()


def setup_context(ctx, inputs, outputs) -> Tensor:
    (x,) = inputs
    ctx.save_for_backward(x)


def backward(ctx, grad):
    (x,) = ctx.saved_tensors
    return grad * x.cos()
