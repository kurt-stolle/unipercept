"""
Simple semantic segmentation model that takes a batched InputData and returns a classification result
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

import unipercept as up

__all__ = ["SemanticSegmenter"]


class SemanticSegmenter(up.model.ModelBase):
    def __init__(self, target_amount: int, input_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, target_amount, kernel_size=3, stride=1, padding=1)

    @override
    def forward(self, inputs: up.model.InputData):
        x = inputs.captures.images[:, 0, :, :, :]
        x = x.float() / 255.0
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x
