"""Baseclass for all backbones."""

# from __future__ import annotations

from __future__ import annotations

import functools
import typing as T
from collections import OrderedDict

import regex as re
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
def query_feature_info(
    module: type[Backbone] | str,
    *args,
    match: str | re.Pattern | T.Iterable[str] | None = None,
    empty_ok: bool = False,
    **kwargs,
) -> BackboneFeatures:
    """
    Utility function to query the feature information of a backbone in configuration
    files.
    """
    from unipercept.config import locate_object

    if isinstance(module, str):
        module = locate_object(module)

    info = module(*args, **kwargs).feature_info
    if match is not None:
        if isinstance(match, str):
            match = re.compile(match)
        if isinstance(match, re.Pattern):
            info = {k: v for k, v in info.items() if match.search(k)}
        elif isinstance(match, T.Iterable):
            match = set(match)
            info = {k: v for k, v in info.items() if k in match}
        else:
            msg = f"Invalid type for {match=} ({type(match)})"
            raise TypeError(msg)

    if len(info) == 0 and not empty_ok:
        args_str = ", ".join(map(str, args))
        kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        msg = f"No features found for {module.__name__}({params}) with {match=}"
        raise ValueError(msg)

    return info
