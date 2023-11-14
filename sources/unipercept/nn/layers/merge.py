from __future__ import annotations

from typing import Mapping, Sequence, cast

import numpy as np
import torch
from detectron2.layers import ShapeSpec
from torch import Tensor, nn
from typing_extensions import override

import unipercept.nn.layers.conv as convolution

__all__ = ["SemanticMerge"]


class SemanticMerge(nn.Module):
    """
    Mergers a multi-level feature pyramid into a single feature map.
    Adapted from SemanticFPN.
    """

    def __init__(
        self,
        in_features: Sequence[str],
        input_shape: Mapping[str, ShapeSpec],
        common_stride: int,
        out_channels: int,
        groups: int = 32,
    ):
        from einops.layers.torch import Rearrange

        super().__init__()

        self.in_features = list(in_features)
        self.common_stride = common_stride

        feature_strides = {k: cast(int, v.stride) for k, v in input_shape.items()}
        feature_channels = {k: cast(int, v.channels) for k, v in input_shape.items()}

        self.scale_heads = cast(list[nn.Module], nn.ModuleList())
        for in_feature in self.in_features:
            head_ops = nn.Sequential()
            head_length = max(
                1,
                int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride)),
            )
            for n in range(head_length):
                in_channels = feature_channels[in_feature] if n == 0 else out_channels
                assert in_channels is not None
                conv = convolution.Conv2d.with_norm_activation(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=groups,
                    norm=lambda ch: nn.GroupNorm(groups, ch, eps=1e-6),
                    activation=lambda: nn.GELU(),
                )

                head_ops.add_module(f"conv_{n}", conv)

                if groups > 1:
                    shf = Rearrange("b (c1 c2) h w -> b (c2 c1) h w", c1=groups)
                    head_ops.add_module(f"shf_{n}", shf)

                if feature_strides[in_feature] != self.common_stride:
                    ups = nn.Upsample(
                        scale_factor=2,
                        mode="bilinear",
                        align_corners=False,
                    )
                    head_ops.add_module(f"ups_{n}", ups)
            self.scale_heads.append(head_ops)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        from timm.layers import trunc_normal_

        with torch.no_grad():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @override
    def forward(self, features: dict[str, Tensor]) -> Tensor:
        x: list[Tensor] = [head(features[self.in_features[i]]) for i, head in enumerate(self.scale_heads)]
        y = x[0]
        for i in range(1, len(self.scale_heads)):
            y = y + x[i]

        return y
