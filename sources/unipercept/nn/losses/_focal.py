from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss
from typing_extensions import override


class SigmoidFocalLoss(nn.Module):
    alpha: float
    gamma: float

    def __init__(self, alpha: float, gamma: float):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    @override
    def forward(self, x: Tensor, y: Tensor, mask: Tensor | None = None) -> Tensor:
        return sigmoid_focal_loss(x, y, alpha=self.alpha, gamma=self.gamma, reduction="none")
