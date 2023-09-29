from __future__ import annotations

from uniutils.config import infer_project_name
from uniutils.config._lazy import call as L

import unipercept as up

from .multidvps_resnet50 import data, model, trainer

__all__ = ["model", "data", "trainer"]

trainer.config.project_name = infer_project_name(__file__)
model.backbone.base = L(up.modeling.backbones.timm.TimmBackbone)(name="convnext_small")
