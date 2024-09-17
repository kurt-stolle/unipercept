r"""
2D positional embeddings as used in Vision Transformer.
"""

from __future__ import annotations

import math

import torch
import typing_extensions as TX
from einops import einsum
from torch import Tensor, nn


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


@torch.no_grad()
def trigonometric_embed_2d(
    x: Tensor,
    mask: Tensor | None,
    num_pos_feats: int,
    temperature: float,
    normalize: bool,
    scale: float,
) -> Tensor:
    if mask is None:
        mask = torch.zeros(
            (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
        )
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

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


class TrigonometricEmbed2d(TrigonometricEmbedBase):
    r"""
    Trigonometric positional embedding for 2D inputs.
    """

    @TX.override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return trigonometric_embed_2d(
            x, mask, self.num_pos_feats, self.temperature, self.normalize, self.scale
        )


@torch.no_grad()
def trigonometric_embed_3d(
    x: Tensor,
    mask: Tensor | None,
    num_pos_feats: int,
    temperature: float,
    normalize: bool,
    scale: float,
) -> Tensor:
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
    if normalize:
        eps = 1e-6
        z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * scale
        y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    dim_t_z = torch.arange((num_pos_feats * 2), dtype=torch.float32, device=x.device)
    dim_t_z = temperature ** (2 * (dim_t_z // 2) / (num_pos_feats * 2))

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


class TrigonometricEmbed3d(TrigonometricEmbedBase):
    r"""
    Trigonometric positional embedding for 3D/2+1D inputs.
    """

    @TX.override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return trigonometric_embed_3d(
            x, mask, self.num_pos_feats, self.temperature, self.normalize, self.scale
        )


@torch.no_grad()
def trigonometric_embed_time(
    x: Tensor,
    mask: Tensor | None,
    num_pos_feats: int,
    temperature: float,
    normalize: bool,
    scale: float,
) -> Tensor:
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
    if normalize:
        eps = 1e-6
        t_embed = t_embed / (t_embed[-1:, :] + eps) * scale

    dim_t = torch.arange(num_pos_feats * 2, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_t = t_embed[:, :, None] / dim_t
    pos_t = torch.stack(
        (pos_t[:, :, 0::2].sin(), pos_t[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    return pos_t


class TrigonometricEmbedTime(nn.Module):
    r"""
    Trigonometric positional embedding for time-oriented inputs.
    """

    @TX.override
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return trigonometric_embed_time(
            x, mask, self.num_pos_feats, self.temperature, self.normalize, self.scale
        )


@torch.no_grad()
def fourier_embed_2d(
    x: Tensor,
    max_freq: float | int | None,
    dim: int,
    use_cos: bool = False,
    use_log: bool = False,
) -> Tensor:
    assert x.dim() == 4, "Input tensor must be 4-dimensional (B, C, H, W)"

    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype
    num_bands = round(dim / (2 * C)) if use_cos else round(dim / C)
    if max_freq is None:
        max_freq = max(H, W) / 2
    max_freq = float(max_freq)

    if use_log:
        s = torch.linspace(
            0.0,
            math.log2(max_freq),
            num_bands,
            device=device,
            dtype=dtype,
        )
        s = 2.0**s
    else:
        s = torch.linspace(1.0, max_freq / 2, num_bands, device=device, dtype=dtype)
    s = s * torch.pi
    x = einsum(x, s, "b c h w, s -> b c s h w").reshape(B, -1, H, W)
    if use_cos:
        x = torch.cat([x.sin(), x.cos()], dim=1)
    else:
        x = x.sin()
    x = torch.nn.functional.pad(
        x, (0, 0, 0, 0, 0, dim - x.shape[1]), mode="constant", value=0
    )
    return x


class FourierEmbed2d(nn.Module):
    r"""
    Use a Fourier basis to embed 2D positions.
    """

    def __init__(
        self,
        dim: int = 512,
        use_cos: bool = False,
        use_log: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.use_cos = use_cos
        self.use_log = use_log

    @TX.override
    def forward(self, x: Tensor, max_freq: float | int | None = None) -> Tensor:
        return fourier_embed_2d(x, max_freq, self.dim, self.use_cos, self.use_log)
