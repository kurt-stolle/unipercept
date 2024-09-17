"""
This package reverts our FPN structure to the old FPN implementation defined in Torchvision.

The original implementation can be found here:
    https://pytorch.org/vision/stable/_modules/torchvision/ops/feature_pyramid_network.html
"""

import math
import typing as T
from collections import OrderedDict

import torch
import torch.nn.functional as F
import typing_extensions as TX
from fvcore.nn.weight_init import c2_xavier_fill
from torch import nn

from unipercept.nn._args import to_2tuple
from unipercept.nn.activations import ActivationSpec
from unipercept.nn.backbones._base import (
    Backbone,
    BackboneFeatureInfo,
    BackboneFeatures,
)
from unipercept.nn.layers import conv, weight
from unipercept.nn.layers.squeeze_excite import SqueezeExcite2d
from unipercept.nn.wrappers import freeze_parameters
from unipercept.utils.inspect import locate_object

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

    out_levels: T.Final[int]

    def forward(
        self,
        x: list[torch.Tensor],
        y: list[torch.Tensor],
    ) -> list[torch.Tensor]:
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

    in_features: T.Final[list[str]]
    interpolate_mode: T.Final[str]

    def __init__(
        self,
        bottom_up: Backbone,
        *,
        in_features: T.Iterable[str],
        out_channels: int,
        out_features: list[str] | None = None,
        norm: T.Callable[..., nn.Module] | None = None,
        extra_blocks: ExtraFPNBlock | None = None,
        squeeze_excite: T.Callable[[int], nn.Module] | bool | None = None,
        conv_module: type[conv.Conv2d] | tuple[type[conv.Conv2d], ...] = conv.Conv2d,
        interpolate_mode: T.Literal["nearest", "nearest-exact", "bilinear"] = "nearest",
        activation: ActivationSpec = None,
        **kwargs,
    ):
        # NOTE: The following patches are added for backwards compatability.
        #       Each option may be deprecated in the future.
        if kwargs.pop("freeze", None) is True:
            bottom_up = freeze_parameters(bottom_up)
        if isinstance(squeeze_excite, bool):
            squeeze_excite = SqueezeExcite2d

        super().__init__(**kwargs)

        self.inner_blocks = nn.ModuleDict()
        self.layer_blocks = nn.ModuleDict()
        self.in_features = list(in_features)
        assert len(self.in_features) > 0, "in_features cannot be empty"
        self.interpolate_mode = interpolate_mode

        in_levels = [
            int(math.log2(bottom_up.feature_info[f].stride)) for f in self.in_features
        ]
        out_levels = in_levels
        if extra_blocks is not None:
            out_levels += [
                in_levels[-1] + i + 1 for i in range(extra_blocks.out_levels)
            ]
        self.out_levels = out_levels
        if out_features is None:
            out_features = ["fpn_" + str(i) for i in out_levels]

        self.out_features = out_features
        assert all(
            f[:4] == "fpn_" for f in self.out_features
        ), "Output features must start with 'fpn_'"
        self.out_indices = [
            i
            for i, level in enumerate(out_levels)
            if f"fpn_{level}" in self.out_features
        ]
        assert len(set(self.out_indices)) == len(
            out_features
        ), "Duplicate output indices"

        self.out_channels = out_channels

        conv_module_inner, conv_module_layer = to_2tuple(conv_module)

        if isinstance(conv_module_inner, str):
            conv_module_inner = locate_object(conv_module_inner)
        if isinstance(conv_module_layer, str):
            conv_module_layer = locate_object(conv_module_layer)

        for feature_name in self.in_features:
            feature_info = bottom_up.feature_info[feature_name]

            # Inner block
            inner_conv = conv_module_inner.with_norm(
                feature_info.channels,
                out_channels,
                kernel_size=1,
                padding=0,
                norm=norm,
            )
            weight.init_xavier_fill_(inner_conv)

            if squeeze_excite is not None:
                inner_block_module = nn.Sequential(
                    squeeze_excite(feature_info.channels), inner_conv
                )

            else:
                inner_block_module = inner_conv
            self.inner_blocks[feature_name] = inner_block_module

            # Layer block
            layer_block_module = conv_module_layer.with_norm_activation(
                out_channels,
                out_channels,
                kernel_size=3,
                norm=norm,
                padding=1,
                activation=activation,
            )
            weight.init_xavier_fill_(layer_block_module)

            self.layer_blocks[feature_name] = layer_block_module

        # Extra outputs
        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise TypeError(
                    f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}"
                )

        self.extra_blocks = extra_blocks
        self.bottom_up = bottom_up

    def get_backbone_features(self) -> BackboneFeatures:
        """
        Returns a ``BackboneFeatures`` mapping for the outputs of this FPN.
        """
        return {
            name: BackboneFeatureInfo(
                channels=self.out_channels,
                stride=2**level,
            )
            for name, level in zip(self.out_features, self.out_levels, strict=True)
        }

    def _forward_fpn(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Run the feature extractor
        features = self.bottom_up(inputs)

        # Define results as a list to allow for easier insertion
        results: list[torch.Tensor] = []

        # Memory to store the previous level
        x_mem: torch.Tensor | None = None

        # Iterate over all feature maps in reverse order
        for name in reversed(self.in_features):
            # Select feature map at named nevel
            f_in = features[name]
            inner = self.inner_blocks[name]
            layer = self.layer_blocks[name]

            # Compute inner mapping (feature to fpn channels)
            x = inner(f_in)

            # Upscale and merge previous level (if exists)
            if x_mem is not None:
                x_size = x.shape[-2:]
                x_ups = F.interpolate(x_mem, size=x_size, mode=self.interpolate_mode)
                x_mem = x + x_ups
            else:
                x_mem = x

            # Forward to next layer
            y = layer(x_mem)

            # Insert result
            results.insert(0, y)

        if self.extra_blocks is not None:
            results.extend(self.extra_blocks(results))

        return OrderedDict(
            [
                (k, results[i])
                for k, i in zip(self.out_features, self.out_indices, strict=True)
            ]
        )

    @TX.override
    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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

        return self._forward_fpn(x)


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d (not actual max_pool2d, we just subsample) on top of the last feature map
    """

    def __init__(self):
        super().__init__()

        self.out_levels = 1

    def forward(
        self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        y = F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0)
        return [y]


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    if T.TYPE_CHECKING:
        # Backwards compatability
        def __init__(self, channels: int): ...

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

            self.out_levels = 2

            self.p6 = conv.Conv2d(channels, channels, 3, stride=2, padding=1)
            self.p7 = conv.Conv2d(channels, channels, 3, stride=2, padding=1)

            for module in [self.p6, self.p7]:
                c2_xavier_fill(module)

    @TX.override
    def forward(
        self,
        x: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        p6 = self.p6(x[-1])
        p7 = self.p7(F.relu(p6))

        return [p6, p7]
