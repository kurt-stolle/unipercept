"""
Implements a feature extraction backbone using Timm.

See: https://huggingface.co/docs/timm/main/en/feature_extraction
"""


from __future__ import annotations

import typing as T

import timm
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing_extensions import override

from .wrapper import (
    ORDER_CHW,
    ORDER_HWC,
    DimensionOrder,
    WrapperBase,
    infer_feature_info,
)

__all__ = ["TimmBackbone", "list_available", "get_dimension_order"]


# ---------------------------- #
# Recipes for common backbones #
# ---------------------------- #
_DIMENSIONORDERS: dict[str, DimensionOrder] = {
    "^resnet.*": ORDER_CHW,
    "^resnest.*": ORDER_CHW,
    "^swin.*": ORDER_HWC,
    "^convnext.*": ORDER_CHW,
    "^efficientnet.*": ORDER_CHW,
    "^mobilenet.*": ORDER_CHW,
}


def get_dimension_order(name: str) -> DimensionOrder:
    import re

    for pattern, order in _DIMENSIONORDERS.items():
        if re.match(pattern, name) is not None:
            return order
    raise ValueError(f"Could not determine dimension order for backbone '{name}'")


# --------------------- #
# Timm backbone wrapper #
# --------------------- #
class TimmBackbone(WrapperBase):
    """Use a (pretrained) model from the `timm` library as a feature extractor, and apply a Feature Pyramid Network on top."""

    def __init__(
        self,
        name: str,
        *,
        pretrained: bool = True,
        dimension_order: str | DimensionOrder | None = None,
        nodes: T.Sequence[int] | None | int = None,
        keys: T.Sequence[str] | None = None,
        **kwargs,
    ):
        if dimension_order is None:
            dims = get_dimension_order(name)
        else:
            dims = DimensionOrder(dimension_order)

        extractor = build_extractor(name, pretrained=pretrained, out_indices=nodes)
        info = infer_feature_info(extractor, dims)

        if keys is None:
            keys = tuple(f"ext.{i}" for i in range(1, len(info) + 1))
        else:
            assert len(keys) == len(info), f"Expected {len(info)} keys, got {len(keys)}"

        super().__init__(dimension_order=dims, feature_info={k: v for k, v in zip(keys, info)}, **kwargs)

        self.ext = extractor

    @override
    def forward_extract(self, images: torch.Tensor) -> TensorDict:
        return TensorDict(
            {k: v for k, v in zip(self.feature_info.keys(), self.ext(images))},
            batch_size=images.shape[:1],
            device=images.device,
        )


# ----------------- #
# Utility functions #
# ----------------- #
def list_available(query: str | None = None) -> list[str]:
    models = timm.list_models(pretrained=True)
    if query is None:
        models.sort()
    else:
        import difflib

        models = difflib.get_close_matches(query, models, n=10, cutoff=0.25)
    return models


def build_extractor(
    name: str,
    *,
    pretrained,
    out_indices: T.Sequence[int] | None | int,
) -> nn.Module:
    mdl = timm.create_model(name, features_only=False, pretrained=pretrained)

    if out_indices is None:
        idxs = tuple(range(len(mdl.feature_info)))
    elif isinstance(out_indices, int):
        num_features = len(mdl.feature_info)
        assert out_indices <= num_features, f"out_indices must be less than or equal to {num_features}"
        idxs = tuple(range(num_features - out_indices, num_features))
    else:
        idxs = tuple(out_indices)

    return torch.jit.script(timm.models.FeatureGraphNet(mdl, out_indices=idxs))

    ## TODO: This is the old code, which is not compatible with the new timm version
    # m = timm.create_model(name, features_only=True, pretrained=pretrained)
    # return m
    ## TODO: This is the old code, which is not compatible with the new timm version
    # m = timm.create_model(name, features_only=True, pretrained=pretrained)
    # return m
