from __future__ import annotations

import enum
import typing as T

import numpy as np
import torch
from torch import Tensor, nn
from typing_extensions import override

import unipercept.nn.layers.conv as convolution
from unipercept.nn.backbones import BackboneFeatureInfo
from unipercept.nn.layers import SqueezeExcite2d
from unipercept.nn.layers.norm import GroupNorm32, NormSpec

__all__ = ["SemanticMerge"]


class WeightMethod(enum.Enum):
    SUM = "sum"
    ATTENTION = "attn"
    FAST_ATTENTION = "fastattn"


class SemanticMerge(nn.Module):
    """
    Mergers a multi-level feature pyramid into a single feature map.
    Adapted from SemanticFPN.
    """

    weight_method: T.Final[WeightMethod]
    common_stride: T.Final[int]
    in_features: T.Final[T.List[str]]

    def __init__(
        self,
        in_features: T.Mapping[str, BackboneFeatureInfo],
        common_stride: int,
        out_channels: int,
        norm: NormSpec = GroupNorm32,
        weight_method: WeightMethod | str = WeightMethod.SUM,
        squeeze_excite: bool = False,
    ):
        super().__init__()

        self.in_features = list(in_features.keys())
        self.common_stride = int(common_stride)

        if isinstance(weight_method, str):
            weight_method = WeightMethod(weight_method)
        self.weight_method = weight_method

        feature_strides = {k: T.cast(int, v.stride) for k, v in in_features.items()}
        feature_channels = {k: T.cast(int, v.channels) for k, v in in_features.items()}

        self.scale_heads = nn.ModuleList()
        for in_feature in self.in_features:
            head_ops = nn.Sequential()
            head_length = max(
                1,
                int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)),
            )
            for n in range(head_length):
                in_channels = feature_channels[in_feature] if n == 0 else out_channels
                assert in_channels is not None

                if squeeze_excite:
                    se = SqueezeExcite2d(in_channels)
                    head_ops.add_module(f"se_{n}", se)

                conv = convolution.Separable2d.with_norm_activation(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=norm,
                    bias=norm is None,
                    activation=nn.GELU,
                )

                head_ops.add_module(f"conv_{n}", conv)

                if feature_strides[in_feature] != self.common_stride:
                    ups = nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    )
                    head_ops.add_module(f"ups_{n}", ups)
            self.scale_heads.append(head_ops)

        if self.weight_method in (WeightMethod.ATTENTION, WeightMethod.FAST_ATTENTION):
            self.edge_weights = nn.Parameter(
                torch.ones(len(self.in_features)), requires_grad=True
            )  # WSM
        else:
            self.edge_weights = None

    def __len__(self) -> int:
        return len(self.in_offsets)

    @override
    def forward(self, features: dict[str, Tensor]) -> Tensor:
        nodes: list[Tensor] = [
            head(features[self.in_features[i]])
            for i, head in enumerate(self.scale_heads)
        ]
        dtype = nodes[0].dtype  # TODO: check whether this is needed

        # Weighting per edge
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
                    (nodes[i] * edge_weights[i]) / (weights_sum + 1e-4)
                    for i in range(len(nodes))
                ],
                dim=-1,
            )
        elif self.weight_method == WeightMethod.SUM:
            assert self.edge_weights is None
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError(f"unknown weight_method {self.weight_method}")

        return torch.sum(out, dim=-1)


class SemanticShuffle(nn.Module):
    """
    Mergers a multi-level feature pyramid into a single feature map. Uses a shuffle operation instead of bilinear
    upsampling.
    """

    weight_method: T.Final[WeightMethod]
    common_stride: T.Final[int]
    in_features: T.Final[T.List[str]]

    def __init__(
        self,
        in_features: T.Iterable[str],
        input_shape: T.Mapping[str, BackboneFeatureInfo],
        common_stride: int,
        out_channels: int,
        norm: NormSpec | None = GroupNorm32,
        weight_method: WeightMethod | str = WeightMethod.SUM,
    ):
        super().__init__()

        self.in_features = list(in_features)
        self.common_stride = int(common_stride)

        if isinstance(weight_method, str):
            weight_method = WeightMethod(weight_method)
        self.weight_method = weight_method

        feature_strides = {k: T.cast(int, v.stride) for k, v in input_shape.items()}
        feature_channels = {k: T.cast(int, v.channels) for k, v in input_shape.items()}

        self.scale_heads = nn.ModuleList()
        for in_feature in self.in_features:
            head_ops = nn.Sequential()
            head_length = max(
                1,
                int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)),
            )
            for n in range(head_length):
                in_channels = feature_channels[in_feature] if n == 0 else out_channels
                assert in_channels is not None

                if feature_strides[in_feature] != self.common_stride:
                    scale_factor = 2
                    expansion = 1
                else:
                    scale_factor = 1
                    expansion = 2

                conv = convolution.Separable2d.with_norm_activation(
                    in_channels,
                    out_channels * scale_factor**2,
                    expansion=expansion,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=norm is None,
                    norm=norm,
                    activation=nn.GELU,
                )

                head_ops.add_module(f"conv_{n}", conv)
                if scale_factor > 0:
                    head_ops.add_module(f"shuf_{n}", nn.PixelShuffle(scale_factor))
            self.scale_heads.append(head_ops)

        if self.weight_method in (WeightMethod.ATTENTION, WeightMethod.FAST_ATTENTION):
            self.edge_weights = nn.Parameter(
                torch.ones(len(self.in_features)), requires_grad=True
            )  # WSM
        else:
            self.edge_weights = None

    def __len__(self) -> int:
        return len(self.in_offsets)

    @override
    def forward(self, features: dict[str, Tensor]) -> Tensor:
        nodes: list[Tensor] = [
            head(features[self.in_features[i]])
            for i, head in enumerate(self.scale_heads)
        ]
        dtype = nodes[0].dtype  # TODO: check whether this is needed

        # Weighting per edge
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
                    (nodes[i] * edge_weights[i]) / (weights_sum + 1e-4)
                    for i in range(len(nodes))
                ],
                dim=-1,
            )
        elif self.weight_method == WeightMethod.SUM:
            assert self.edge_weights is None
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError(f"unknown weight_method {self.weight_method}")

        return torch.sum(out, dim=-1)
