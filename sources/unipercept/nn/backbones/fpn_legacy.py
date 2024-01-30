"""Implements a generalized Feature Pyramid Network, with strong defaults."""

# from __future__ import annotations

from __future__ import annotations

import enum
import itertools
import typing as T

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import override

import unipercept.nn.layers as _ML
from unipercept.log import get_logger
from unipercept.nn.layers.activation import ActivationSpec
from unipercept.nn.layers.conv import Conv2d as ConvModule
from unipercept.nn.layers.norm import NormSpec

from ._base import Backbone, BackboneFeatureInfo, BackboneFeatures

__all__ = [
    "FeaturePyramidNetwork",
    "FeaturePyramidBackbone",
    "WeightMethod",
    "NodeConfig",
    "Routing",
    "build_default_routing",
    "build_pan_routing",
    "build_quad_routing",
]

# _DEFAULT_ACTIVATION = functools.partial(nn.ReLU, inplace=True)
_DEFAULT_ACTIVATION = nn.GELU
_DEFAULT_NORM = nn.BatchNorm2d

_logger = get_logger(__name__)


class NodeConfig(T.TypedDict):
    """
    A dictionary that specifies the inputs and weight method for a feature pyramid network (FPN) node.

    Attributes:
        level (int): The level of the feature map, from the perspective of the backbone.
        in_offsets (List[int]): The offsets of the input feature maps.
        weight_method (str): The method used to weight the input feature maps.
    """

    level: int
    in_offsets: list[int]
    weight_method: str


class Routing(T.TypedDict):
    num_levels: int
    nodes: T.Sequence[NodeConfig]


class _Resample(nn.Sequential):
    """Resample a feature map to a different resolution."""

    in_channels: T.Final[int]
    out_channels: T.Final[int]
    in_stride: T.Final[int]
    out_stride: T.Final[int]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        in_stride: int,
        out_stride: int,
        downsample_mode: str | None,
        upsample_mode: str | None,
        norm: NormSpec | None = _DEFAULT_NORM,
        pre_activation: bool | ActivationSpec = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_stride = in_stride
        self.out_stride = out_stride

        scale: float = out_stride / in_stride

        pre_act_mod: ActivationSpec | None
        if pre_activation is True:
            pre_act_mod = _DEFAULT_ACTIVATION
        elif pre_activation is False:
            pre_act_mod = None
        elif callable(pre_activation):
            pre_act_mod = pre_activation
        else:
            raise ValueError(
                f"pre_activation must be bool or callable, got {pre_activation}"
            )

        if scale > 1 and downsample_mode == "conv":
            kernel_size = int(scale) + 1
            # dilation = int(scale / 2)
            stride = int(scale)
            self.add_module(
                "conv_downsample",
                _ML.conv.Separable2d.with_norm_activation(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    # dilation=dilation,
                    padding=kernel_size // 2,
                    norm=norm,
                    activation=pre_act_mod,
                ),
            )

            scale = 1.0  # such that the next downsample is skipped
        elif in_channels != out_channels:
            self.add_module(
                "conv",
                _ML.conv.Conv2d.with_norm_activation(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding="same",
                    norm=norm,
                    activation=pre_act_mod,
                ),
            )

        if scale > 1:  # e.g. 2 / 1
            if downsample_mode is None:
                raise ValueError("downsample_mode must be specified when downsampling")
            if downsample_mode in ("max", "avg"):
                down_cls = _ML.conv.utils.POOLING_LAYERS[downsample_mode]
                down_mod = down_cls(
                    kernel_size=int(scale) + 1, stride=int(scale), padding="same"
                )
            else:
                assert isinstance(downsample_mode, str), type(downsample_mode)
                down_mod = _ML.Interpolate2d(
                    scale_factor=1 / scale, mode=downsample_mode
                )
            self.add_module("downsample", down_mod)
        elif scale < 1:  # e.g. 2 / 1
            if upsample_mode is None:
                raise ValueError("upsample_mode must be specified when upsampling")
            assert isinstance(upsample_mode, str), type(upsample_mode)
            up_mod = _ML.Interpolate2d(scale_factor=1 / scale, mode=upsample_mode)
            self.add_module("upsample", up_mod)
        else:
            pass


class WeightMethod(enum.StrEnum):
    SUM = "sum"
    ATTENTION = "attn"
    FAST_ATTENTION = "fastattn"


class _Combine(nn.Module):
    r"""
    Combine multiple FPN levels into a single level.
    """

    weight_method: T.Final[WeightMethod]
    in_offsets: T.Final[T.Tuple[int, ...]]
    out_channels: T.Final[int]
    out_stride: T.Final[int]

    def __init__(
        self,
        info: T.Sequence[BackboneFeatureInfo],
        in_offsets: T.Sequence[int],
        out_channels: int,
        out_stride: int,
        *,
        downsample_mode: str,
        upsample_mode: str,
        norm: NormSpec | None,
        weight_method: WeightMethod | str = WeightMethod.FAST_ATTENTION,
    ):
        super().__init__()
        self.in_offsets = tuple(i for i in in_offsets)
        self.out_channels = out_channels
        self.out_stride = out_stride
        self.weight_method = WeightMethod(weight_method)

        self.resample = nn.ModuleList()
        for i in self.in_offsets:
            in_channels_level = info[i].channels
            in_stride_level = info[i].stride
            self.resample.append(
                _Resample(
                    in_channels_level,
                    out_channels,
                    in_stride=in_stride_level,
                    out_stride=out_stride,
                    downsample_mode=downsample_mode,
                    upsample_mode=upsample_mode,
                    norm=norm,
                )
            )

        if self.weight_method in (WeightMethod.ATTENTION, WeightMethod.FAST_ATTENTION):
            self.edge_weights = nn.Parameter(
                torch.ones(len(self)), requires_grad=True
            )  # WSM
        else:
            self.edge_weights = None

    def __len__(self) -> int:
        return len(self.in_offsets)

    @override
    def forward(self, x: T.List[torch.Tensor]) -> torch.Tensor:
        # Resample
        nodes = []
        for i, resample in zip(self.in_offsets, self.resample):
            n = x[i]
            n = resample(n)
            nodes.append(n)

        # Weighting per edge
        dtype = x[0].dtype
        if self.weight_method == WeightMethod.ATTENTION:
            assert self.edge_weights is not None
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == WeightMethod.FAST_ATTENTION:
            assert self.edge_weights is not None
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [
                    (nodes[i] * edge_weights[i]) / (weights_sum + 0.0001)
                    for i in range(len(nodes))
                ],
                dim=-1,
            )
        elif self.weight_method == WeightMethod.SUM:
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError(f"unknown weight_method {self.weight_method}")

        return torch.sum(out, dim=-1)


class _Node(nn.Module):
    r"""
    A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """

    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super().__init__()
        self.combine = combine
        self.after_combine = after_combine

    @override
    def forward(self, x: T.List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class FeaturePyramidNetworkLayer(nn.Module):
    def __init__(
        self,
        info_list: T.Sequence[BackboneFeatureInfo],
        nodes: T.Sequence[NodeConfig],
        out_channels: int,
        *,
        level_to_stride: dict[int, int],
        num_levels: int,
        activation: ActivationSpec,
        norm: NormSpec,
        conv_module: type[ConvModule],
        norm_combine=True,
        padding="same",
        downsample_mode: str = "max",
        upsample_mode: str = "bilinear",
    ):
        super().__init__()
        self.num_levels = num_levels
        info_list = list(info_list) + [
            BackboneFeatureInfo(out_channels, level_to_stride[fc["level"]])
            for fc in nodes
        ]

        self.fnode = nn.ModuleList()
        for i, c in enumerate(nodes):
            _logger.debug(f"FPN node {i} : {c}")

            in_offsets = tuple(c["in_offsets"])
            weight_method = WeightMethod(c["weight_method"])
            level = c["level"]

            out_stride = info_list[level].stride

            combine = _Combine(
                info_list,
                in_offsets,
                out_channels,
                out_stride,
                downsample_mode=downsample_mode,
                upsample_mode=upsample_mode,
                norm=norm if norm_combine else None,
                weight_method=weight_method,
            )

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=padding,
                norm=norm,
                activation=activation,
            )

            conv_layer = conv_module.with_norm_activation(**conv_kwargs)
            after_combine.add_module("conv", conv_layer)

            self.fnode.append(_Node(combine=combine, after_combine=after_combine))

        self.feature_info = info_list[-num_levels::]

    @override
    def forward(self, x: T.List[torch.Tensor]):
        for fn in self.fnode:
            y = checkpoint(fn, x) if False else fn(x)
            x.append(y)
        return x[-self.num_levels : :]


class FeaturePyramidNetwork(nn.Module):
    """Generic feature pyramid network."""

    feature_info: T.Final[BackboneFeatures]
    in_features: T.Final[T.Tuple[str, ...]]

    def __init__(
        self,
        feature_info: BackboneFeatures,
        in_features: T.Sequence[str],
        *,
        routing: Routing,
        out_channels: int,
        num_hidden: int = 1,
        norm: NormSpec = _DEFAULT_NORM,
        activation: ActivationSpec = _DEFAULT_ACTIVATION,
        downsample_mode: str = "max",
        upsample_mode: str = "bilinear",
        fill_stride_ratio: float = 2.0,
        conv_module: type[ConvModule] = _ML.conv.Separable2d,
    ):
        super().__init__()

        nodes = routing["nodes"]

        num_levels = routing["num_levels"]
        if num_levels < len(in_features):
            raise ValueError(
                "number of input must be greater than or equal to the number of input features"
            )
        if num_levels <= 0:
            raise ValueError("number of levels must be positive")

        _logger.debug(
            f"buidling FPN ({in_features}) with {num_levels} levels and {num_hidden} cells, using nodes: {nodes}"
        )

        # Add nodes to fill in missing levels
        info_list: list[BackboneFeatureInfo] = [
            feature_info[key] for key in in_features
        ]
        pre_activation = False
        self.resample = nn.ModuleDict()
        for level in range(num_levels):
            if level < len(info_list):
                _logger.debug(f"using FPN level {level} from input features")
                continue
            if level == 0:
                raise ValueError("Cannot add a new level at level 0")
            last_channels = info_list[level - 1].channels
            last_stride = info_list[level - 1].stride
            next_stride = int(last_stride * fill_stride_ratio)

            _logger.debug(f"adding FPN level {level} with stride {next_stride}")

            # Adds a coarser level by downsampling the last feature map
            self.resample[str(level)] = _Resample(
                in_channels=last_channels,
                out_channels=out_channels,
                in_stride=last_stride,
                out_stride=next_stride,
                downsample_mode="max",
                upsample_mode=None,
                pre_activation=pre_activation,
                norm=norm,
            )

            pre_activation = True  # only the first resample has no pre-activation

            info_list.append(
                BackboneFeatureInfo(channels=out_channels, stride=next_stride)
            )

        # Add layers
        self.cell = _ML.SequentialList()

        level_to_stride: dict[int, int] = {
            i: info_list[i].stride for i in range(len(info_list))
        }

        for rep in range(num_hidden):
            _logger.debug(f"Building cell {rep}")
            fpn_layer = FeaturePyramidNetworkLayer(
                info_list=info_list,
                nodes=nodes,
                out_channels=out_channels,
                num_levels=num_levels,
                downsample_mode=downsample_mode,
                upsample_mode=upsample_mode,
                norm=norm,
                activation=activation,
                conv_module=conv_module,
                level_to_stride=level_to_stride,
            )
            self.cell.add_module(str(rep), fpn_layer)
            info_list = fpn_layer.feature_info

        self.in_features = tuple(in_features)

        self.feature_info = {
            f"fpn_{i+1}": info for i, info in enumerate(info_list[-num_levels::])
        }

    @override
    def forward(self, features: T.Dict[str, torch.Tensor]) -> T.Dict[str, torch.Tensor]:
        inputs = [features[key] for key in self.in_features]

        # Add extra levels (if needed)
        for resample in self.resample.values():
            inputs.append(resample(inputs[-1]))

        # FPN stage
        # inputs = checkpoint(self.cell, inputs, use_reentrant=False)
        inputs = self.cell(inputs)

        # Result
        features = {}
        for i, input in enumerate(inputs):
            features[f"fpn_{i+1}"] = input

        return features


def build_default_routing(
    num_levels: int,
    weight_method: WeightMethod | str = WeightMethod.FAST_ATTENTION,
) -> Routing:
    """Default FPN nodes."""
    min_level = 0
    max_level = num_levels - 1
    weight_method = WeightMethod(weight_method)

    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]  # noqa: E731
    level_all_ids = lambda level: node_ids[level]  # noqa: E731
    id_cnt = itertools.count(num_levels)

    nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path.
        nodes.append(
            {
                "level": i,
                "in_offsets": [level_last_id(i), level_last_id(i + 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    for i in range(min_level + 1, max_level + 1):
        # bottom-up path.
        nodes.append(
            {
                "level": i,
                "in_offsets": level_all_ids(i) + [level_last_id(i - 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    return {"nodes": nodes, "num_levels": num_levels}


def build_pan_routing(
    num_levels: int,
    weight_method: WeightMethod | str = WeightMethod.FAST_ATTENTION,
) -> Routing:
    """
    Uses the structure from Path Aggregation Networks.

    Paper: https://arxiv.org/abs/1803.01534
    """

    weight_method = WeightMethod(weight_method)

    min_level = 0
    max_level = num_levels - 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]  # noqa: E731
    id_cnt = itertools.count(num_levels)

    nodes = []
    for i in range(max_level, min_level - 1, -1):
        # top-down path.
        offsets = (
            [level_last_id(i), level_last_id(i + 1)]
            if i != max_level
            else [level_last_id(i)]
        )
        nodes.append(
            {
                "level": i,
                "in_offsets": offsets,
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    for i in range(min_level, max_level + 1):
        # bottom-up path.
        offsets = (
            [level_last_id(i), level_last_id(i - 1)]
            if i != min_level
            else [level_last_id(i)]
        )
        nodes.append(
            {
                "level": i,
                "in_offsets": offsets,
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    return {"num_levels": num_levels, "nodes": nodes}


def build_quad_routing(
    num_levels: int,
    weight_method: WeightMethod | str = WeightMethod.FAST_ATTENTION,
) -> Routing:
    """
    A dynamic quad fpn config that can adapt to different min/max levels.

    It has four paths:
        (up_down -> bottom_up) + (bottom_up -> up_down).

    Paper: https://ieeexplore.ieee.org/document/9225379
    """
    weight_method = WeightMethod(weight_method)
    quad_method = WeightMethod.FAST_ATTENTION

    min_level = 0
    max_level = num_levels - 1
    node_ids = {min_level + i: [i] for i in range(num_levels)}

    level_last_id = lambda level: node_ids[level][-1]  # noqa: E731
    level_all_ids = lambda level: node_ids[level]  # noqa: E731
    level_first_id = lambda level: node_ids[level][0]  # noqa: E731

    id_cnt = itertools.count(num_levels)

    nodes = []
    for i in range(max_level - 1, min_level - 1, -1):
        # top-down path 1.
        nodes.append(
            {
                "level": i,
                "in_offsets": [level_last_id(i), level_last_id(i + 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    for i in range(min_level + 1, max_level):
        # bottom-up path 2.
        nodes.append(
            {
                "level": i,
                "in_offsets": level_all_ids(i) + [level_last_id(i - 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    i = max_level
    nodes.append(
        {
            "level": i,
            "in_offsets": [level_first_id(i)] + [level_last_id(i - 1)],
            "weight_method": weight_method,
        }
    )
    node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(min_level + 1, max_level + 1, 1):
        # bottom-up path 3.
        nodes.append(
            {
                "level": i,
                "in_offsets": [
                    level_first_id(i),
                    level_last_id(i - 1)
                    if i != min_level + 1
                    else level_first_id(i - 1),
                ],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))
    node_ids[min_level].append(node_ids[min_level][-1])

    for i in range(max_level - 1, min_level, -1):
        # top-down path 4.
        nodes.append(
            {
                "level": i,
                "in_offsets": [node_ids[i][0]]
                + [node_ids[i][-1]]
                + [level_last_id(i + 1)],
                "weight_method": weight_method,
            }
        )
        node_ids[i].append(next(id_cnt))
    i = min_level
    nodes.append(
        {
            "level": i,
            "in_offsets": [node_ids[i][0]] + [level_last_id(i + 1)],
            "weight_method": weight_method,
        }
    )
    node_ids[i].append(next(id_cnt))
    node_ids[max_level].append(node_ids[max_level][-1])

    # NOTE: the order of the quad path is reversed from the original, my code expects the output of
    # each FPN repeat to be same as input from backbone, in order of increasing reductions
    for i in range(min_level, max_level + 1):
        # quad-add path.
        nodes.append(
            {
                "level": i,
                "in_offsets": [node_ids[i][2], node_ids[i][4]],
                "weight_method": quad_method,
            }
        )
        node_ids[i].append(next(id_cnt))

    return {"nodes": nodes, "num_levels": num_levels}


class FeaturePyramidBackbone(Backbone):
    """A backbone that uses a feature pyramid network (FPN) to extract features."""

    base: Backbone
    fpn: FeaturePyramidNetwork

    def __init__(
        self,
        base: Backbone,
        freeze: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        base : Backbone
            The backbone to use.
        kwargs
            Additional arguments to pass to the FeaturePyramidNetwork.
        """

        fpn = FeaturePyramidNetwork(base.feature_info, **kwargs)

        super().__init__(feature_info=fpn.feature_info)

        self.base = base
        if freeze:
            for p in self.base.parameters():
                p.requires_grad_(False)
        self.fpn = fpn

    @override
    def forward(self, images: torch.Tensor) -> T.Dict[str, torch.Tensor]:
        features = self.base(images)
        features = self.fpn(features)
        return features
