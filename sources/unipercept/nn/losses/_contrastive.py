"""
Utilities for contrastive loss functions
"""

from __future__ import annotations

import torch
import torch.nn as nn

from unipercept.nn.losses.mixins import ScaledLossMixin

__all__ = ["cosine_distance", "TripletMarginSimilarityLoss"]


class TripletMarginSimilarityLoss(ScaledLossMixin, nn.TripletMarginWithDistanceLoss):
    """
    Implements a triplet contrastive loss using cosine distance between features, i.e. promoting that the target and
    anchor have a higher cosine similarity than the target and negative.
    """

    def __init__(self, *, margin: float = 0.1, **kwargs):
        super().__init__(**kwargs, margin=margin, distance_function=cosine_distance)

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        loss = nn.functional.triplet_margin_with_distance_loss(
            anchor,
            positive,
            negative,
            distance_function=self.distance_function,
            margin=self.margin,
            swap=self.swap,
            reduction=self.reduction,
        )
        return self.scale * loss


@torch.jit.script
def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1 - nn.functional.cosine_similarity(x, y)
