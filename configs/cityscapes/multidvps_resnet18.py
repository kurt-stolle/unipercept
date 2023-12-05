"""
Debugging config for MultiDVPS on Cityscapes with ResNet-18 backbone, used for testing and development purposes.
"""


from __future__ import annotations


from unimodels import multidvps
from unipercept.config import get_session_name, call as L, bind as B
from .multidvps_resnet50 import model, engine, data

import unipercept as up

__all__ = ["model", "engine", "data"]

engine.params.session_name = get_session_name(__file__)
engine.params.eval_steps = 100
engine.params.save_steps = 100
engine.params.logging_steps = 50
engine.params.gradient_accumulation_steps=1

model.backbone.bottom_up = L(up.nn.backbones.timm.TimmBackbone)(name="resnet18d")
model.backbone.out_channels = 24
model.detector.localizer.encoder.out_channels = 64
model.feature_encoder.merger.out_channels = 64
model.feature_encoder.heads[multidvps.KEY_MASK].out_channels = 64
model.feature_encoder.heads[multidvps.KEY_DEPTH].out_channels = 32
model.kernel_mapper.input_dims = 32
model.kernel_mapper.attention_heads = 2
model.kernel_mapper.mapping[multidvps.KEY_REID].out_channels = 16
