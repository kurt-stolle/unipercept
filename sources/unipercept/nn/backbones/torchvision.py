"""Implements a feature extraction backbone using Torchvision."""

from __future__ import annotations

import difflib
import typing as T
import warnings
from collections import OrderedDict
from typing import override

import regex as re
import torch
import torchvision.models
import torchvision.models.feature_extraction

from unipercept.log import logger

from ._wrapper import (
    ORDER_CHW,
    ORDER_HWC,
    DimensionOrder,
    WrapperBase,
    catalog,
    infer_feature_info,
)

# ---------------------------- #
# Recipes for common backbones #
# ---------------------------- #

_EXTRACTION_NODES: OrderedDict[re.Pattern, re.Pattern | list[str]] = OrderedDict()
_EXTRACTION_NODES.update(
    # Common defaults
    {
        re.compile(R"^resnet\d+.*"): [
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ],
        re.compile(R"^swin.*"): [
            # "features.0",
            "features.1",
            #    "features.2",
            "features.3",
            #    "features.4",
            "features.5",
            #    "features.6",
            "features.7",
        ],
    }
)
_EXTRACTION_NODES.update(
    # Fallback to regular expression
    {
        re.compile(R"^resn.*"): re.compile(r"^layer\d+"),
        re.compile(R"^convnext.*"): re.compile(r"^conv\d+"),
        re.compile(R"^efficientnet.*"): re.compile(r"^blocks\.\d+"),
        re.compile(R"^mobilenet.*"): re.compile(r"^features\.\d+"),
    }
)


def get_default_nodes(name: str) -> list[str]:
    for name_re, nodes in _EXTRACTION_NODES.items():
        if name_re.match(name) is None:
            continue
        if isinstance(nodes, T.Sequence):
            return nodes
        assert isinstance(nodes, re.Pattern)
        nodes_train, nodes_eval = TorchvisionBackbone.list_nodes(name)
        if set(nodes_train) != set(nodes_eval):
            # TODO (@kurt-stolle) can this be supported?
            msg = (
                f"Feature output nodes of backbone '{name}' ({name_re.pattern}) differ between training "
                "and evaluation mode. Cannot automatically determine node names"
            )
        result = []
        for key in nodes_train:
            match = nodes.search(key)
            if match is None:
                continue
            node = match.group()
            if node in result:
                continue
            result.append(node)
        if len(result) == 0:
            msg = (
                f"Feature output nodes of backbone '{name}' ({name_re.pattern}) do not match the expected "
                f"pattern '{nodes.pattern}' ({name_re.pattern}). Please specify them manually.\n\nNodes: {nodes_train}"
            )
            raise ValueError(msg)
        return result

    msg = f"Feature output nodes of backbone '{name}' are unknown. Please specify them manually."
    raise ValueError(msg)


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


@catalog.register("torchvision")
class TorchvisionBackbone(WrapperBase):
    r"""
    Wrapper for Torchvision backbones.

    See Also
    --------
    - `Results <https://pytorch.org/vision/stable/models.html>`_
    - `Feature Extraction <https://pytorch.org/vision/stable/feature_extraction.html>`_
    """

    def __init__(
        self,
        name: str,
        *,
        weights: torchvision.models.WeightEnum | str | None = None,
        nodes: T.Mapping[str, str] | None = None,
        dimension_order: str | DimensionOrder | None = None,
        jit_script: bool = False,
        requires_grad: bool = True,
        **kwargs,
    ):
        if nodes is None:
            nodes = {k.replace(".", "_"): k for k in get_default_nodes(name)}
        if dimension_order is None:
            dims = get_dimension_order(name)
        else:
            dims = DimensionOrder(dimension_order)

        if isinstance(weights, str):
            weights = torchvision.models.get_weight(weights)
        elif weights is None:
            weights = torchvision.models.get_model_weights(name)

        self.model_weights = weights

        extractor = build_extractor(name, nodes=nodes, weights=weights)
        info = infer_feature_info(extractor, dims)

        super().__init__(
            name,
            dimension_order=dims,
            feature_info=info,
            **kwargs,
        )

        if not requires_grad:
            for p in extractor.parameters():
                p.requires_grad = False
        if jit_script:
            try:
                extractor = torch.jit.script(extractor)  # type: ignore
            except Exception as e:
                logger.warn(f"Failed to script the backbone: {e}")

        self.ext = extractor

    @override
    def forward_extract(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return self.ext(images)

    @classmethod
    @T.override
    def list_nodes(cls, name: str) -> tuple[list[str], list[str]]:
        model = torchvision.models.get_model(name, weights=None)
        return torchvision.models.feature_extraction.get_graph_node_names(model)

    @classmethod
    @T.override
    def list_models(
        cls, query: str | None = None, pretrained: bool = False, **kwargs
    ) -> list[str]:
        if pretrained:
            warnings.warn("Pretrained queries are not supported yet")
            return []

        models = torchvision.models.list_models(module=torchvision.models)
        if query is None:
            models.sort()
        else:
            models = difflib.get_close_matches(query, models, n=10, cutoff=0.25)
        return models

    @T.override
    def extra_repr(self) -> str:
        weights = self.model_weights
        if hasattr(weights, "value"):
            weights = weights.value  # type: ignore
        return f"model_weights={weights}, {super().extra_repr()}"


def build_extractor(
    name: str, *, nodes: T.Mapping[str, str], weights: str | None = None
):
    from unipercept.config.env import get_env

    model = torchvision.models.get_model(name, weights=weights)

    if get_env(bool, "UP_NN_BACKBONES_DISABLE_GRAPH", default=False):
        msg = "Torchvision backbone extraction is not yet supported without graph trace mode"
        raise NotImplementedError(msg)

    return torchvision.models.feature_extraction.create_feature_extractor(
        model, {v: k for k, v in nodes.items()}
    )


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
