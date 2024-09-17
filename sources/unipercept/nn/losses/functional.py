"""
Implements common functional operations for losses.
"""

from __future__ import annotations

import torch
from torch import Tensor


def split_into_patches(
    x: Tensor,
    sizes: tuple[int, int],
    strides: tuple[int, int] | None = None,
) -> Tensor:
    r"""
    Splits tensor into N, p_height x p_width blocks.
    """

    if strides is None:
        strides = (sizes[0], sizes[1])

    # Parse shape as [..., H, W]
    *dots, _, _ = x.shape
    for dim, (size, stride) in enumerate(zip(sizes, strides, strict=False)):
        x = x.unfold(len(dots) + dim, size, stride)

    # ... x P x Hp x Wp
    return x.reshape(*dots, -1, sizes[0], sizes[1])


def reduce_batch(image_loss: Tensor, mask: Tensor) -> Tensor:
    divisor = torch.sum(mask)
    if divisor == 0:
        return 0
    return torch.sum(image_loss) / divisor


def reduce_batch(image_loss: Tensor, mask: Tensor) -> Tensor:
    valid = mask.nonzero()
    image_loss[valid] = image_loss[valid] / mask[valid]

    return torch.mean(image_loss)
