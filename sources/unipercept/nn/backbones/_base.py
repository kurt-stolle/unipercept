"""Baseclass for all backbones."""

# from __future__ import annotations

from __future__ import annotations

import typing as T
from collections import OrderedDict
import functools
import torch
import torch.nn as nn

__all__ = ["Backbone", "BackboneFeatureInfo", "BackboneFeatures", "query_feature_info"]

# ---------------- #
# Feature metadata #
# ---------------- #


class BackboneFeatureInfo(T.NamedTuple):
    """
    Information about a feature.

    Properties
    ----------
    channels : int
        The number of channels of the feature.
    stride : int
        The stride of the feature (with respect to the input image).
    """

    channels: int
    stride: int


BackboneFeatures: T.TypeAlias = T.Dict[str, BackboneFeatureInfo]


# -------------- #
# Base interface #
# -------------- #


class Backbone(nn.Module):
    """
    Baseclass for backbones.
    """

    feature_info: T.Final[BackboneFeatures]

    def __init__(self, *, feature_info: BackboneFeatures, **kwargs):
        super().__init__(**kwargs)

        self.feature_info = {k.replace(".", "_"): v for k, v in feature_info.items()}

    def forward(self, images: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        """
        Extract features from the input data.

        Parameters
        ----------
        images : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The (processed) image, e.g. with added padding or resized, as required for the backbone.
        list[torch.Tensor]
            The extracted features.
        """
        raise NotImplementedError()


@functools.lru_cache()
def query_feature_info(module: type[Backbone], *args, **kwargs):
    """
    Utility function to query the feature information of a backbone in confiugration
    files.
    """
    bb = module(*args, **kwargs)
    return bb.feature_info
