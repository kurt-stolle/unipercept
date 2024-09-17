"""
Various extensions of ``torch.signal``.
"""

from __future__ import annotations

import typing as T

import torch.signal.windows


def get_gaussian_2d(
    kernel_size: T.Sequence[int] | tuple[int, int],
    sigma: T.Sequence[float] | tuple[float, float],
    dtype: torch.dtype,
    device: torch.device | None = None,
    sym: bool = True,
) -> torch.Tensor:
    """
    Computes a 2D gaussian kernel.
    """
    with torch.no_grad():
        k_x = torch.signal.windows.gaussian(
            kernel_size[0], std=sigma[0], sym=sym, dtype=dtype, device=device
        )
        k_y = torch.signal.windows.gaussian(
            kernel_size[1], std=sigma[1], sym=sym, dtype=dtype, device=device
        )

        k_max = torch.maximum(k_x.max(), k_y.max())
        k = k_y.unsqueeze(-1) * k_x

        # normalize such that the maximum value of k is equal to k_max
        return (k / k.max()) * k_max
