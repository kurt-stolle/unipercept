"""
This package reverts our FPN structure to the old FPN implementation defined in Torchvision.

The original implementation can be found here: 
    https://pytorch.org/vision/stable/_modules/torchvision/ops/feature_pyramid_network.html
"""

from __future__ import annotations

import typing as T
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import typing_extensions as TX

from unipercept.nn.backbones._base import Backbone
from unipercept.nn.layers import SqueezeExcite2d, conv

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

    in_features: torch.jit.Final[T.List[str]]

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

        self.inner_blocks = nn.ModuleDict()
        self.layer_blocks = nn.ModuleDict()
        self.in_features = list(in_features)

        for feature_name in self.in_features:
            feature_info = bottom_up.feature_info[feature_name]

            # Inner block
            if squeeze_excite:
                inner_block_module = nn.Sequential(
                    SqueezeExcite2d(feature_info.channels),
                    conv.Conv2d.with_norm(
                        feature_info.channels,
                        out_channels,
                        kernel_size=1,
                        padding=0,
                        norm=norm,
                    ),
                )
            else:
                inner_block_module = conv.Conv2d.with_norm(
                    feature_info.channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    norm=norm,
                )
            self.inner_blocks[feature_name] = inner_block_module

            # Layer block
            layer_block_module = conv.Separable2d.with_norm(
                out_channels, out_channels, kernel_size=3, norm=norm, padding=1
            )
            self.layer_blocks[feature_name] = layer_block_module

        # Do not TX.override initialization of backbone and extra blocks.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Extra outputs
        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise TypeError(
                    f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}"
                )

        self.extra_blocks = extra_blocks

        # Extrator
        self.bottom_up = bottom_up
        if freeze:
            for p in self.bottom_up.parameters():
                p.requires_grad_(False)

    @TX.override
    def forward(self, inputs: T.Dict[str, torch.Tensor]) -> T.Dict[str, torch.Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Parameters
        ----------
        inputs
            feature maps for each feature level.

        Returns
        -------
        dict[str, torch.Tensor]
            Feature maps after FPN layers.
        """

        # Run the feature extractor
        features = self.bottom_up(inputs)

        # Define results as a list to allow for easier insertion
        results: T.List[torch.Tensor] = []

        # Memory to store the previous level
        x_mem: torch.Tensor | None = None

        # Iterate over all feature maps in reverse order
        for name in reversed(self.in_features):
            # Select feature map at named nevel
            f_in = features[name]

            # Compute inner mapping (feature to fpn channels)
            x = self.inner_blocks[name](f_in)

            # Upscale and merge previous level (if exists)
            if x_mem is not None:
                x_size = x.shape[-2:]
                x_ups = F.interpolate(x_mem, size=x_size, mode="nearest-exact")
                x_mem = x + x_ups
            else:
                x_mem = x

            # Forward to next layer
            y = self.layer_blocks[name](x_mem)

            # Insert result
            results.insert(0, y)

        if self.extra_blocks is not None:
            results.extend(self.extra_blocks(results))

        # make it back an OrderedDict
        out = OrderedDict([(f"fpn_{i+1}", v) for i, v in enumerate(results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d (not actual max_pool2d, we just subsample) on top of the last feature map
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
    ) -> T.List[torch.Tensor]:
        y = F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0)
        return [y]


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    if T.TYPE_CHECKING:
        # Backwards compatability
        def __init__(self, channels: int):
            ...

    else:

        def __init__(self, channels=None, **kwargs):
            super().__init__()

            if channels is None:
                try:
                    in_channels = kwargs["in_channels"]
                    out_channels = kwargs["out_channels"]
                except KeyError:
                    raise ValueError("channels must be provided")
                assert in_channels == out_channels
                channels = in_channels

            self.p6 = conv.Conv2d(channels, channels, 3, stride=2, padding=1)
            self.p7 = conv.Conv2d(channels, channels, 3, stride=2, padding=1)

            for module in [self.p6, self.p7]:
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)

    @TX.override
    def forward(
        self,
        x: T.List[torch.Tensor],
    ) -> T.List[torch.Tensor]:
        p6 = self.p6(x[-1])
        p7 = self.p7(F.gelu(p6))

        return [p6, p7]
