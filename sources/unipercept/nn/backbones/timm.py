"""
Implements a feature extraction backbone using Timm.

See: https://huggingface.co/docs/timm/main/en/feature_extraction
"""

import difflib
import operator
import typing as T
from collections import OrderedDict
from typing import override

import timm
import timm.data
import torch
import torch.fx
from torch import nn
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

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
_DIMENSION_ORDER_MAP: dict[str, DimensionOrder] = {
    "^resnet.*": ORDER_CHW,
    "^resnest.*": ORDER_CHW,
    "^swin.*": ORDER_HWC,
    "^convnext.*": ORDER_CHW,
    "^efficientnet.*": ORDER_CHW,
    "^mobilenet.*": ORDER_CHW,
}


def _get_dimension_order(name: str) -> DimensionOrder:
    import re

    for pattern, order in _DIMENSION_ORDER_MAP.items():
        if re.match(pattern, name) is not None:
            return order
    raise ValueError(f"Could not determine dimension order for backbone '{name}'")


# --------------------- #
# Timm backbone wrapper #
# --------------------- #
@catalog.register("timm")
class TimmBackbone(WrapperBase):
    r"""
    Wrapper for Timm (Pytorch IMage Models) backbones.

    See Also
    --------
    - `Project page <https://huggingface.co/docs/timm>`_
    - `Results <https://huggingface.co/docs/timm/results>`_
    """

    def __init__(
        self,
        name: str,
        *,
        pretrained: bool = True,
        nodes: T.Sequence[str] | T.Mapping[str, str | int] | None = None,
        requires_grad: bool = True,
        jit_script: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name : str
            The name of the backbone model.
        pretrained : bool, optional
            Whether to use a pretrained model.
        nodes : Sequence[str] | Mapping[str, Union[str, int]], optional
            The names of the nodes to extract, or a mapping from output keys to nodes names/indices.
            When None, the default nodes are used.
        requires_grad : bool, optional
            Whether to require gradients for the backbone.
        """
        extractor, nodes, config = _build_extractor(
            name, pretrained=pretrained, nodes=nodes, use_sync_batchnorm=False
        )
        dims = _get_dimension_order(name)
        info = infer_feature_info(extractor, dims)
        assert len(set(info.keys()) - set(nodes.keys())) == 0, (
            info.keys(),
            nodes.keys(),
        )

        if "mean" in config:
            kwargs.setdefault("mean", config["mean"])
        if "std" in config:
            kwargs.setdefault("std", config["std"])

        self.pretrained = pretrained

        super().__init__(
            name,
            dimension_order=dims,
            feature_info=info,
            **kwargs,
        )

        if not requires_grad:
            for p in extractor.parameters():
                p.requires_grad = False
            self.eval()
        if jit_script:
            extractor = torch.jit.script(extractor)
        self.ext = extractor

    @override
    def forward_extract(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return self.ext(images)
        # output_nodes = OrderedDict()
        # for k, v in zip(self.feature_info.keys(), self.ext(images)):
        #    output_nodes[k] = v
        # return output_nodes

    @classmethod
    def list_nodes(cls, name: str) -> tuple[list[str], list[str]]:
        model = timm.create_model(name, features_only=False, pretrained=False)
        return get_graph_node_names(model)

    @classmethod
    def list_available(
        cls, query: str | None = None, pretrained: bool = False, **kwargs
    ) -> list[str]:
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
            models = difflib.get_close_matches(query, models, n=10, cutoff=0.25)
        return models

    @T.override
    def extra_repr(self) -> str:
        if self.pretrained:
            return f"{super().extra_repr()}, pretrained"
        return super().extra_repr()


def _build_extractor(
    name: str,
    *,
    pretrained,
    nodes: T.Sequence[str] | T.Mapping[str, str | int] | None = None,
    use_sync_batchnorm: bool = False,
) -> tuple[nn.Module | torch.fx.GraphModule, dict[str, str], dict[str, T.Any]]:
    mdl = timm.create_model(
        name, features_only=False, pretrained=pretrained, scriptable=True
    )

    # Convert all BatchNorm to SyncBatchNorm
    if use_sync_batchnorm:
        if hasattr(mdl, "convert_sync_batchnorm"):
            mdl = mdl.convert_sync_batchnorm()
        else:
            mdl = nn.SyncBatchNorm.convert_sync_batchnorm(mdl)

    # Query the configuration metadata
    config = timm.data.resolve_data_config({}, model=mdl)

    # Defaults suggested by Timm and all nodes in the graph
    avail_defaults = list(map(operator.itemgetter("module"), mdl.feature_info))
    avail_nodes = list(get_graph_node_names(mdl))

    # Construct mapping of feature name -> node name to extract
    node_map = {}
    if nodes is None:
        node_map.update({m: m for m in avail_defaults})
    elif isinstance(nodes, T.Mapping):
        for k, v in nodes.items():
            if isinstance(v, int):
                v = avail_defaults[v]
            node_map[str(k)] = str(v)
    elif isinstance(nodes, T.Sequence):
        node_map.update({str(i): str(i) for i in nodes})

    if len(node_map) == 0:
        msg = f"No node_map specified for {name!r}. Available node_map: {avail_defaults} and {avail_nodes}."
        raise ValueError(msg)

    for k, v in node_map.items():
        if v in avail_defaults:
            continue
        if v in avail_nodes:
            continue
        msg = (
            f"Node {v!r} is not available in {name!r}. Choose from default features "
            f"{avail_defaults} or node_map {avail_nodes}."
        )
        raise ValueError(msg)

    # Torchvision expects keys and values to be swithced
    return_nodes = {v: k for k, v in node_map.items()}
    assert len(return_nodes) > 0, return_nodes

    # Create graph
    logger.debug(
        f"Creating Timm feature extractor backbone {name!r} with nodes {return_nodes!r}"
    )

    gm = create_feature_extractor(mdl, return_nodes=return_nodes)
    # gm = timm.models.FeatureGraphNet(mdl, out_indices=idxs, return_dict=True)
    return gm, node_map, config
