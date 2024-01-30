"""
Implements a feature extraction backbone using Timm.

See: https://huggingface.co/docs/timm/main/en/feature_extraction
"""


from __future__ import annotations

import typing as T
from collections import OrderedDict

import timm
import timm.data
import torch
import torch.nn as nn
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


def _get_dimension_order(name: str) -> DimensionOrder:
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
        nodes: T.Sequence[int] | None | int = None,
        keys: T.Sequence[str] | None = None,
        use_graph: bool = True,
        **kwargs,
    ):
        dims = _get_dimension_order(name)
        extractor, config = _build_extractor(
            name, pretrained=pretrained, out_indices=nodes, use_graph=use_graph
        )
        info = infer_feature_info(extractor, dims)

        if "mean" in config:
            kwargs.setdefault("mean", config["mean"])
        if "std" in config:
            kwargs.setdefault("std", config["std"])

        if keys is None:
            keys = tuple(f"ext_{i}" for i in range(1, len(info) + 1))
        else:
            assert len(keys) == len(info), f"Expected {len(info)} keys, got {len(keys)}"

        super().__init__(
            dimension_order=dims,
            feature_info={k: v for k, v in zip(keys, info)},
            **kwargs,
        )

        self.ext = extractor

    @override
    def forward_extract(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        output_nodes = OrderedDict()
        for k, v in zip(self.feature_info.keys(), self.ext(images)):
            output_nodes[k] = v
        return output_nodes


# ----------------- #
# Utility functions #
# ----------------- #
def list_available(query: str | None = None, pretrained: bool = False) -> list[str]:
    """
    Lists available backbones from the `timm` library.

    Parameters
    ----------
    query : str, optional
        If specified, only backbones containing this string will be returned.
    pretrained : bool, optional
        If True, only pretrained backbones will be returned.

    Returns
    -------
    list[str]
        List of available backbones.

    """
    models = timm.list_models(pretrained=pretrained)
    if query is None:
        models.sort()
    else:
        import difflib

        models = difflib.get_close_matches(query, models, n=10, cutoff=0.25)
    return models


def _build_extractor(
    name: str,
    *,
    pretrained,
    out_indices: T.Sequence[int] | None | int,
    use_graph: bool,
) -> tuple[nn.Module, dict[str, T.Any]]:
    mdl = timm.create_model(name, features_only=False, pretrained=pretrained)
    config = timm.data.resolve_data_config({}, model=mdl)

    if out_indices is None:
        idxs = tuple(range(len(mdl.feature_info)))
    elif isinstance(out_indices, int):
        num_features = len(mdl.feature_info)
        assert (
            out_indices <= num_features
        ), f"out_indices must be less than or equal to {num_features}"
        idxs = tuple(range(num_features - out_indices, num_features))
    else:
        idxs = tuple(out_indices)

    if use_graph:
        out = timm.models.FeatureGraphNet(mdl, out_indices=idxs)
    else:
        out = timm.models.FeatureListNet(mdl, out_indices=idxs)
    return out, config

    ## TODO: This is the old code, which is not compatible with the new timm version
    # m = timm.create_model(name, features_only=True, pretrained=pretrained)
    # return m
    ## TODO: This is the old code, which is not compatible with the new timm version
    # m = timm.create_model(name, features_only=True, pretrained=pretrained)
    # return m
