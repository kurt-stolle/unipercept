"""
Simple classification model that takes a batched InputData and returns a classification result
"""

import torch.nn as nn
from typing_extensions import override

import unipercept as up


class Classifier(up.model.ModelBase):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    @override
    def forward(self, inputs: up.model.InputData):
        x = inputs.captures.images[:, 0, :, :, :]
        x.float()
        x /= 255.0

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out
