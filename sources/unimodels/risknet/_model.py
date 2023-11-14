import typing as T

import torch
import torch.nn as nn
from typing_extensions import override

import unipercept as up

__all__ = ["RiskNet"]


class RiskNet(nn.Module):
    def __init__(
        self,
        *,
        backbone: up.nn.backbones.Backbone,
        example_layer: nn.Module,
        example_loss: nn.Module,
    ):
        super().__init__()

        self.backbone = backbone
        self.example_layer = example_layer
        self.example_loss = example_loss

    @override
    def forward(self, inputs: up.tensors.InputData) -> T.Dict[str, torch.Tensor] | T.List[T.Dict[str, T.Any]]:
        backbone_outputs = self.backbone(inputs.captures.images)

        example_fpn_reduce = torch.stack(
            [self.example_layer(v).mean(dim=(-2, -1)) for v in backbone_outputs.values()]
        ).mean(dim=1)

        if self.training:
            return self.forward_training(example_fpn_reduce)
        else:
            return self.forward_inference(example_fpn_reduce)

    def forward_training(self, logits) -> T.Dict[str, torch.Tensor]:
        loss = self.example_loss(logits)

        return {
            "loss_example": loss,
        }

    def forward_inference(self, logits) -> list[T.Dict[str, torch.Tensor]]:
        return [
            {
                "riskyness": f.mean(),
            }
            for f in logits
        ]
