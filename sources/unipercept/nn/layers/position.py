"""
Positional encoding modules.
"""
import math

import torch
from torch import nn, Tensor


class TrigonometricEmbed2d(nn.Module):
    """
    2D positional encoding using sine and cosine functions.
    """

    def __init__(
        self,
        channels,
        *,
        temperature: int = 10000,
        scale: float | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert channels % 4 == 0, "Number of channels must be divisible by 4"

        self.channels = channels // 4
        self.temperature = temperature
        self.eps = eps
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        assert x.shape[-3] == self.channels * 4, "Number of channels must match"
        assert x.ndim == 4, "Input tensor must have 4 dimensions (B, C, H, W)"
        if mask is None:
            inv_mask = torch.ones(x.shape[-3:], device=x.device, dtype=torch.bool)
        else:
            inv_mask = ~mask
        y_embed = inv_mask.cumsum(1, dtype=torch.float32)
        x_embed = inv_mask.cumsum(2, dtype=torch.float32)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.channels, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.channels)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        return (
            torch.cat((pos_y, pos_x), dim=3)
            .permute(0, 3, 1, 2)
            .flatten(0, 1)
            .unsqueeze(0)
        )
