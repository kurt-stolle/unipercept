r"""
2D positional embeddings as used in Vision Transformer.
"""

from __future__ import annotations

import math

import torch
import typing_extensions as TX
from torch import nn, Tensor


class TrigonometricEmbedBase(nn.Module):
    r"""
    Trigonometric positional embedding base class. Does not implement the forward method.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, **kwargs
    ):
        if scale is not None and normalize is False:
            msg = "Agument {normalize=} should be True for {scale=}"
            raise ValueError(msg)
        if scale is None:
            scale = 2 * math.pi

        super().__init__(**kwargs)

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    @TX.override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        msg = f"Method {self.__class__.__name__}.forward() is not implemented"
        raise NotImplementedError(msg)


class TrigonometricEmbed2d(TrigonometricEmbedBase):
    r"""
    Trigonometric positional embedding for 2D inputs.
    """

    @TX.override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TrigonometricEmbed3d(TrigonometricEmbedBase):
    r"""
    Trigonometric positional embedding for 3D/2+1D inputs.
    """

    @TX.override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # b, t, c, h, w
        assert (
            x.dim() == 5
        ), f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(1), x.size(3), x.size(4)),
                device=x.device,
                dtype=torch.bool,
            )
        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        dim_t_z = torch.arange(
            (self.num_pos_feats * 2), dtype=torch.float32, device=x.device
        )
        dim_t_z = self.temperature ** (2 * (dim_t_z // 2) / (self.num_pos_feats * 2))

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos_z = torch.stack(
            (pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5
        ).flatten(4)
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z).permute(
            0, 1, 4, 2, 3
        )  # b, t, c, h, w
        return pos


class TrigonometricEmbedTime(nn.Module):
    r"""
    Trigonometric positional embedding for time-oriented inputs.
    """

    @TX.override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        # t, bs, c
        assert (
            x.dim() == 3
        ), f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        if mask is None:
            mask = torch.zeros(
                (
                    x.size(0),
                    x.size(1),
                ),
                device=x.device,
                dtype=torch.bool,
            )
        not_mask = ~mask
        t_embed = not_mask.cumsum(0, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            t_embed = t_embed / (t_embed[-1:, :] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats * 2, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_t = t_embed[:, :, None] / dim_t
        pos_t = torch.stack(
            (pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        return pos_t
