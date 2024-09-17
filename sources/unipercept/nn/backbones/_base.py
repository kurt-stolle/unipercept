"""Baseclass for all backbones."""

import copy
import functools
import typing as T
import warnings

import regex as re
from torch import nn

from unipercept.types import Tensor
from unipercept.utils.catalog import CatalogFromPackageMetadata
from unipercept.utils.inspect import locate_object

__all__ = [
    "Backbone",
    "BackboneFeatureInfo",
    "BackboneFeatures",
    "query_feature_info",
    "check_feature_info",
    "catalog",
]

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


type BackboneFeatures = dict[str, BackboneFeatureInfo]


# -------------- #
# Base interface #
# -------------- #

type BackboneResult = dict[str, Tensor]


class Backbone(nn.Module):
    """
    Baseclass for backbones.
    """

    feature_info: T.Final[BackboneFeatures]

    def __init__(
        self,
        *,
        feature_info: BackboneFeatures,
        ignore_invalid_info: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not check_feature_info(feature_info, raises=not ignore_invalid_info):
            warnings.warn(
                "Invalid feature info, ignoring because keyword argument ignore_invalid_info is True!",
                RuntimeWarning,
                stacklevel=2,
            )
        self.feature_info = copy.deepcopy(feature_info)

    @T.override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.feature_info})"

    def forward(self, images: Tensor) -> BackboneResult:
        """
        Extract features from the input data.

        Parameters
        ----------
        images : Tensor
            The input data.

        Returns
        -------
        Tensor
            The (processed) image, e.g. with added padding or resized, as required for the backbone.
        list[Tensor]
            The extracted features.
        """
        raise NotImplementedError()

    if T.TYPE_CHECKING:
        __call__ = forward


# -------------------- #
# Feature info helpers #
# -------------------- #

_ILLEGAL_FEATURE_CHARS = re.compile(r"[\.]")


def check_feature_info(info: T.Any, *, raises=True) -> T.TypeGuard[BackboneFeatures]:
    r"""
    Check whether the backbone (which could be external, timm/torchvision) has
    returned a valid feature info object.
    """

    if not isinstance(info, dict):
        if raises:
            msg = f"Invalid type for {info=} ({type(info)})"
            raise TypeError(msg)
        return False
    if len(info) == 0:
        if raises:
            msg = f"Empty {info=}"
            raise ValueError(msg)
        return False

    for k, v in info.items():
        if not isinstance(k, str):
            if raises:
                msg = f"Key {k!r} ({type(k)}) is not a string.\n\n{info=}"
                raise TypeError(msg)
            return False
        if _ILLEGAL_FEATURE_CHARS.search(k):
            if raises:
                msg = f"Key {k!r} contains illegal characters from {_ILLEGAL_FEATURE_CHARS.pattern}.\n\n{info=}"
                raise ValueError(msg)
            return False
        if not isinstance(v, BackboneFeatureInfo):
            if raises:
                msg = f"Invalid type for {v=} ({type(v)}), expected {BackboneFeatureInfo}.\n\n{info=}"
                raise TypeError(msg)
            return False
        if v.channels <= 1:
            if raises:
                msg = f"{v.channels=} <= 1!\n\n{info=}"
                raise ValueError(msg)
            return False
        if v.stride <= 1:
            if raises:
                msg = f"Invalid value for {v.stride=} <= 1\n\n{info=}"
                raise ValueError(msg)
            return False
    return True


@functools.lru_cache
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

    Notes
    -----
    Users are encouraged to use partial functions or factory methods instead of calling
    this function during module initialization!
    """

    if isinstance(module, str):
        module = locate_object(module)
    mod = module(*args, **kwargs)
    info = mod.feature_info
    del mod
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


# ------------------- #
# Catalog registry #
# ------------------- #


catalog: CatalogFromPackageMetadata[Backbone, T.Mapping[str, str]] = (
    CatalogFromPackageMetadata(group="unipercept.backbones")
)
