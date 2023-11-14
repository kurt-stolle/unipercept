"""
Implements the feature branch modules.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override

__all__ = ["FeatureSelector", "FeatureEncoder"]


class FeatureSelector(nn.Module):
    """
    Selects a feature map from the given feature maps dict.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @override
    def forward(self, feats: TensorDictBase) -> torch.Tensor:
        return feats.get(self.name)


class FeatureEncoder(nn.Module):
    """
    Encodes the given feature maps into multiple feature embeddings.
    """

    def __init__(self, merger: nn.Module, heads: dict[str, nn.Module], shared_encoder: T.Optional[nn.Module] = None):
        super().__init__()

        self.merger = merger
        self.shared_encoder = shared_encoder
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

        if self.shared_encoder is not None:
            merged = self.shared_encoder(merged)

        fe_emb = {key: head(merged) for key, head in self.heads.items()}

        return TensorDict(
            fe_emb,
            batch_size=[merged.shape[0]],
            device=merged.device,
        )
