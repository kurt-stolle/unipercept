"""Implements a feature extraction backbone using Torchvision."""


from __future__ import annotations

import typing as T
import warnings
from collections import OrderedDict

import torch
from typing_extensions import override

from .wrapper import (
    ORDER_CHW,
    ORDER_HWC,
    DimensionOrder,
    WrapperBase,
    infer_feature_info,
)

__all__ = [
    "TorchvisionBackbone",
    "list_available",
    "list_nodes",
    "build_extractor",
    "get_dimension_order",
]

# ---------------------------- #
# Recipes for common backbones #
# ---------------------------- #

_EXTRACTION_NODES: dict[str, tuple[str, ...]] = {
    "^resnet.*": tuple(f"layer{i}" for i in range(1, 5)),
    "^swin_v2.*": (
        "features.1.1.add_1",
        "features.3.1.add_1",
        "features.5.9.add_1",
        "features.7.1.add_1",
    ),
}


def get_default_nodes(name: str) -> tuple[str, ...]:
    import re

    for pattern, nodes in _EXTRACTION_NODES.items():
        if re.match(pattern, name) is not None:
            return nodes
    raise ValueError(f"Could not determine extraction node names for '{name}'")


_DIMENSIONORDERS: dict[str, DimensionOrder] = {
    r"^resnet.*": ORDER_CHW,
    r"^swin.*": ORDER_HWC,
    r"^convnext.*": ORDER_CHW,
}


def get_dimension_order(name: str) -> DimensionOrder:
    import re

    for pattern, order in _DIMENSIONORDERS.items():
        if re.match(pattern, name) is not None:
            return DimensionOrder(order)
    raise ValueError(f"Could not determine dimension order for '{name}'")


# ---------------------------- #
# Torchvision backbone wrapper #
# ---------------------------- #


class TorchvisionBackbone(WrapperBase):
    def __init__(
        self,
        name: str,
        *,
        weights: str | None = "DEFAULT",
        # out_channels: int,
        # extra_blocks: ExtraFPNBlock,
        nodes: T.Sequence[str] | None = None,
        keys: T.Sequence[str] | None = None,
        dimension_order: str | DimensionOrder | None = None,
        jit_script: bool = False,
        **kwargs,
    ):
        if nodes is None:
            nodes = get_default_nodes(name)
        if dimension_order is None:
            dims = get_dimension_order(name)
        else:
            dims = DimensionOrder(dimension_order)

        extractor = build_extractor(name, nodes=nodes, weights=weights)

        info = infer_feature_info(extractor, dims)
        if keys is None:
            keys = [f"ext.{i}" for i in range(1, len(info) + 1)]
        else:
            assert len(keys) == len(info), f"Expected {len(info)} keys, got {len(keys)}"

        super().__init__(
            dimension_order=dims,
            feature_info={k: v for k, v in zip(keys, info)},
            **kwargs,
        )

        self.ext = extractor

        if jit_script:
            self.ext = torch.jit.script(self.ext)

    @override
    def forward_extract(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return self.ext(images)


# ----------------- #
# Utility functions #
# ----------------- #
def list_available(query: str | None = None, pretrained: bool = False) -> list[str]:
    import torchvision.models

    if pretrained:
        warnings.warn("Pretrained queries are not supported yet")
        return []

    models = torchvision.models.list_models(module=torchvision.models)
    if query is None:
        models.sort()
    else:
        import difflib

        models = difflib.get_close_matches(query, models, n=10, cutoff=0.25)
    return models


def list_nodes(name: str) -> set[str]:
    from torchvision.models import get_model
    from torchvision.models.feature_extraction import get_graph_node_names

    model = get_model(name, weights=None)
    train_nodes, eval_nodes = get_graph_node_names(model)

    return set(train_nodes) & set(eval_nodes)


def build_extractor(name: str, *, nodes: T.Iterable[str], weights: str | None = None):
    from torchvision.models import get_model
    from torchvision.models.feature_extraction import create_feature_extractor

    model = get_model(name, weights=weights)

    return create_feature_extractor(model, list(nodes))


# --- #
# CLI #
# --- #
# if __name__ == "__main__":
#     # Quick and simple CLI for listing available models and nodes of those models
#     import argparse

#     # Create argment parser. Main commands are:
#     # 1. list               : list available models
#     # 2. model <model> ...  : (see below)
#     #        shapes         : list available weights for a given model name
#     #        nodes          : list available nodes for a given model name
#     root_p = argparse.ArgumentParser()
#     root_s = root_p.add_subparsers(dest="command")

#     # Command: list
#     list_p = root_s.add_parser("list")
#     list_p.add_argument("--query", "-q", type=str, default=None)

#     # Command: model
#     model_p = root_s.add_parser("model")
#     model_p.add_argument("name", type=str)
#     model_s = model_p.add_subparsers(dest="sub_command")

#     # Command: model <name> shapes
#     shapes_p = model_s.add_parser("shapes")
#     # p_shapes.add_argument("--weights", "-w", type=str, default=None)

#     # Command: model <name> nodes
#     p_nodes = model_s.add_parser("nodes")

#     # Parse arguments
#     args = root_p.parse_args()

#     # Handle commands
#     if args.command == "list":
#         models = list_available(args.query)
#         print("\n".join(models))
#     elif args.command == "model":
#         if args.sub_command == "nodes":
#             nodes = list_nodes(args.name)
#             print("\n".join(nodes))
#         else:
#             raise RuntimeError(f"Unknown subcommand: {args.sub_command}")
#     else:
#         raise RuntimeError(f"Unknown command: {args.command}")
