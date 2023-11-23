from __future__ import annotations

import unipercept as up
from unipercept.utils.config import call as L
from unipercept.utils.config import get_project_name, get_session_name

from .multidvps_resnet50 import data, model, trainer

__all__ = ["model", "data", "trainer"]

trainer.config.session_name = get_session_name(__file__)
model.backbone.bottom_up = L(up.nn.backbones.timm.TimmBackbone)(name="convnext_small")
model.backbone.in_features = ["ext.1", "ext.2", "ext.3", "ext.4"]
