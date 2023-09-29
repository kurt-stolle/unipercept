"""
Implements the feature branch modules.
"""

from __future__ import annotations

import typing as T

import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override

__all__ = ["FeatureEncoder"]


class FeatureEncoder(nn.Module):
    def __init__(self, merger: nn.Module, heads: dict[str, nn.Module], encoder: T.Optional[nn.Module] = None):
        super().__init__()

        self.merger = merger
        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)

    def keys(self) -> T.Sequence[str]:
        return tuple(self.heads.keys())

    @override
    def forward(self, feats: TensorDictBase) -> TensorDict:
        """
        Args:
            feat (torch.Tensor): A feature map of shape (B, C, H, W).
        """

        merged = self.merger(feats)

        if self.encoder is not None:
            merged = self.encoder(merged)

        fe_emb = {key: head(merged) for key, head in self.heads.items()}

        return TensorDict(
            fe_emb,
            batch_size=[merged.shape[0]],
            device=merged.device,
        )
