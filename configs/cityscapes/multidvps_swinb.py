from __future__ import annotations

import unipercept as up
from unipercept.config import get_session_name
from unipercept.config._lazy import call as L

from .multidvps_resnet50 import data, model, engine

__all__ = ["model", "data", "engine"]

engine.params.session_name = get_session_name(__file__)
model.backbone.base = L(up.nn.backbones.timm.TimmBackbone)(name="swin_base_patch4_window12_384")
