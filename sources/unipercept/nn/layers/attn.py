"""
Generic attention and variants.
"""

from __future__ import annotations

import typing as T

import torch
import typing_extensions as TX
from torch import Tensor, nn


class Attention(nn.Module):
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        h_dim: int | None = None,
        *,
        heads: int = 8,
        gate: bool = True,
        proj: nn.Module | T.Callable[[int, int], nn.Module] = nn.Linear,
    ):
        super().__init__()

        if h_dim is None:
            assert q_dim % heads == 0, (q_dim, heads)
            h_dim = q_dim // heads

        hidden_dim = h_dim * heads
        self.heads = heads

        self.proj_q = proj(q_dim, hidden_dim)
        self.proj_k = proj(k_dim, hidden_dim)
        self.proj_v = proj(v_dim, hidden_dim)
        self.proj_o = proj(hidden_dim, q_dim)
        if gate:
            self.proj_g = proj(q_dim, hidden_dim)
        else:
            self.register_module("proj_g", None)
        self.scale = h_dim**-0.5  # 1/sqrt(head_dim)

    @TX.override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        bias: Tensor | None = None,
    ) -> Tensor:
        r"""
        Parameters
        ----------
        q
            Query tensor of shape (..., q_dim).
        k
            Key tensor of shape (..., k_dim).
        v
            Value tensor of shape (..., v_dim).
        bias
            Additive bias tensor of shape (..., num_heads, q_len, k_len).

        Returns
        -------
        Tensor
            Output tensor of shape (..., q_dim).
        """

        wq = self.proj_q(q).view(*q.shape[:-1], 1, self.heads, -1) * self.scale
        wk = self.proj_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.heads, -1)
        a = (wq * wk).sum(-1).softmax(-2)

        if bias is not None:
            a = a + bias[..., None]

        wv = self.proj_v(v).view(*v.shape[:-2], 1, v.shape[-2], self.heads, -1)
        o = (a[..., None] * wv).sum(-3)
        o = o.view(*o.shape[:-2], -1)

        if self.proj_g is not None:
            g = self.proj_g(q)
            o = torch.sigmoid(g) * o

        return self.proj_o(o)
