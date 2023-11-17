"""
Debugging config for MultiDVPS on Cityscapes with ResNet-18 backbone, used for testing and development purposes.
"""


from __future__ import annotations


from unimodels import multidvps
from unipercept.utils.config import get_session_name, call as L, bind as B
from .multidvps_resnet50 import model, trainer, data

import unipercept as up

__all__ = ["model", "trainer", "data"]

trainer.config.session_name = get_session_name(__file__)
trainer.config.train_epochs = 50
trainer.config.eval_steps = 1000
trainer.config.logging_steps = 10

model.backbone.base = L(up.nn.backbones.timm.TimmBackbone)(name="resnet18")
model.backbone.out_channels = 24
model.detector.localizer.encoder.out_channels = 64
model.feature_encoder.shared_encoder.out_channels=64
model.feature_encoder.heads[multidvps.KEY_MASK].out_channels=64
model.feature_encoder.heads[multidvps.KEY_DEPTH].out_channels=32
model.kernel_mapper.input_dims=32
model.kernel_mapper.mapping[multidvps.KEY_REID].out_channels=32