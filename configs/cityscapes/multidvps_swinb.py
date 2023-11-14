from __future__ import annotations

import unipercept as up
from unipercept.utils.config import get_project_name
from unipercept.utils.config._lazy import call as L

from .multidvps_resnet50 import data, model, trainer

__all__ = ["model", "data", "trainer"]

trainer.config.project_name = get_project_name(__file__)
model.backbone.base = L(up.nn.backbones.timm.TimmBackbone)(name="swin_base_patch4_window12_384")
