"""
Utilities for contrastive loss functions
"""

import torch
import torch.nn as nn

__all__ = ["cosine_distance", "TripletMarginSimilarityLoss"]


class TripletMarginSimilarityLoss(nn.TripletMarginWithDistanceLoss):
    """
    Implements a triplet contrastive loss using cosine distance between features, i.e. promoting that the target and
    anchor have a higher cosine similarity than the target and negative.
    """

    def __init__(self, *, margin: float = 0.1, **kwargs):
        super().__init__(**kwargs, margin=margin, distance_function=cosine_distance)


@torch.jit.script
def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1 - nn.functional.cosine_similarity(x, y)
