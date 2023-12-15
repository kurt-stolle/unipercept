"""
This package reverts our FPN structure to the old FPN implementation defined in Torchvision.

The original implementation can be found here: 
    https://pytorch.org/vision/stable/_modules/torchvision/ops/feature_pyramid_network.html
"""

import enum
import typing as T
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

from ..layers import SqueezeExcite2d, conv
from ._base import Backbone

__all__ = ["FeaturePyramidNetwork", "LastLevelMaxPool", "LastLevelP6P7"]


class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Parameters
    ----------
        results
            the result of the FPN
        x
            the original feature maps

    Returns
    -------
    results
        the extended set of results of the FPN
    """

    def forward(
        self,
        x: T.List[torch.Tensor],
        y: T.List[torch.Tensor],
    ) -> T.List[torch.Tensor]:
        raise NotImplementedError()


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedT.Dict[torch.Tensor], containing
    the feature maps on top of which the FPN will be added.

    Parameters
    ----------
    in_channels_list
        number of channels for each feature map that is passed to the module
    out_channels
        number of channels of the FPN representation
    norm
        Module specifying the normalization layer to use. Default: None
    extra_blocks
        if provided, extra operations will be performed. It is expected to take the fpn features, the original
        features and the names of the original features as input, and returns a new list of feature maps and their
        corresponding names
    """

    in_features: T.List[str]

    def __init__(
        self,
        bottom_up: Backbone,
        in_features: T.Iterable[str],
        out_channels: int,
        norm: T.Optional[T.Callable[..., nn.Module]],
        extra_blocks: T.Optional[ExtraFPNBlock],
        freeze: bool = False,
        squeeze_excite: bool = False,
    ):
        super().__init__()

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        self.in_features = list(in_features)

        for feature_name in self.in_features:
            feature_info = bottom_up.feature_info[feature_name]

            if squeeze_excite:
                inner_block_module = nn.Sequential(
                    SqueezeExcite2d(feature_info.channels),
                    conv.Conv2d.with_norm(feature_info.channels, out_channels, kernel_size=1, padding=0, norm=norm),
                )
            else:
                inner_block_module = conv.Conv2d.with_norm(
                    feature_info.channels, out_channels, kernel_size=1, padding=0, norm=norm
                )
            layer_block_module = conv.Separable2d.with_norm(
                out_channels, out_channels, kernel_size=3, norm=norm, padding=1
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # Do not override initialization of backbone and extra blocks.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise TypeError(f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}")

        self.extra_blocks = extra_blocks
        self.bottom_up = bottom_up
        if freeze:
            for p in self.bottom_up.parameters():
                p.requires_grad_(False)

    def get_result_from_inner_blocks(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    @override
    def forward(self, inputs: torch.Tensor) -> T.Dict[str, torch.Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedT.Dict[torch.Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedT.Dict[torch.Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        """
        # unpack into two lists for easier handling
        features = self.bottom_up(inputs)

        x = [features[k] for k in self.in_features]

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results = self.extra_blocks(results, x)

        # make it back an OrderedDict
        out = OrderedDict([(f"fpn.{i+1}", v) for i, v in enumerate(results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d (not actual max_pool2d, we just subsample) on top of the last feature map
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: T.List[torch.Tensor],
        y: T.List[torch.Tensor],
    ) -> T.List[torch.Tensor]:
        x.append(F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0))
        return x


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = conv.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.p7 = conv.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(
        self,
        p: T.List[torch.Tensor],
        c: T.List[torch.Tensor],
    ) -> T.List[torch.Tensor]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.gelu(p6))
        p.extend([p6, p7])
        return p
