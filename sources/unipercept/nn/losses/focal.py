# from __future__ import annotations
import typing as T
from typing import override

import torch
from torch import Tensor, nn
from torchvision.ops import sigmoid_focal_loss


class SigmoidFocalLoss(nn.Module):
    alpha: T.Final[float]
    gamma: T.Final[float]

    def __init__(self, *, alpha: float, gamma: float, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.gamma = gamma

    @override
    def forward(
        self, x: Tensor, y: Tensor, mask: torch.Optional[torch.Tensor] = None
    ) -> Tensor:
        return sigmoid_focal_loss(
            x, y, alpha=self.alpha, gamma=self.gamma, reduction="none"
        )
