"""Implements the baseclass for a backbone, which is a feature extractor that returns a dict of features."""

import copy
import enum
import typing as T

import torch
from torch import nn
from torch.utils._pytree import tree_map

import unipercept.log as logger
from unipercept.types import Tensor
from unipercept.utils.catalog import CatalogFromPackageMetadata

from ._base import Backbone, BackboneFeatureInfo, BackboneFeatures, BackboneResult
from ._normalize import Normalizer

__all__ = [
    "WrapperBase",
    "DimensionOrder",
    "ORDER_CHW",
    "ORDER_HWC",
    "infer_feature_info",
    "catalog",
]


class DimensionOrder(enum.StrEnum):
    """The format of the extracted features."""

    CHW = enum.auto()
    HWC = enum.auto()


ORDER_CHW = DimensionOrder.CHW
ORDER_HWC = DimensionOrder.HWC

DimensionOrderType: T.TypeAlias = T.Literal["chw", "hwc"] | DimensionOrder


####################################
# Base class for Backbone wrappers #
####################################

_DEFAULT_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_DEFAULT_IMAGENET_STD = (0.229, 0.224, 0.225)


class WrapperPermute(nn.Module):
    """
    Permutes CHW to HWC or vice versa.
    """

    def __init__(self):
        super().__init__()

    @T.override
    def forward(
        self, features: T.OrderedDict[str, Tensor]
    ) -> T.OrderedDict[str, Tensor]:
        return tree_map(hwc_to_chw, features)


class WrapperBase(Backbone):
    """
    Baseclass for a backbone, which is a feature extractor that returns a dict of features.

    Parameters
    ----------
    dimension_order : str | DimensionOrder
        The format of the **extracted features**.
        Note that this is not the format of the input data, which is always assumed to
        be (B, 3, H, W) in RGB format!
    dimension_output : str | DimensionOrder | None
        The format of the **output features**. If None, the output format is the same
        as that provided by the wrapped model. If different than dimension_order, a
        permutation is applied.
    mean : list[float]
        The mean values for each channel, which are used for normalization.
    std : list[float]
        The standard deviation values for each channel, which are used for normalization.
    """

    dimension_order: T.Final[str]  # use str to make PyTorch happy

    def __init__(
        self,
        name: str,
        *,
        mean=_DEFAULT_IMAGENET_MEAN,
        std=_DEFAULT_IMAGENET_STD,
        dimension_order: str | DimensionOrder = "chw",
        dimension_output: str | DimensionOrder | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_name = name
        self.dimension_order = DimensionOrder(dimension_order).value
        if (
            dimension_output is not None
            and DimensionOrder(dimension_output) != dimension_order
        ):
            self.permute = WrapperPermute()
        else:
            self.permute = nn.Identity()

        self.normalize = Normalizer(mean, std)

    @T.override
    def forward(self, images: Tensor) -> BackboneResult:
        """
        Extract features from the input data.

        Parameters
        ----------
        images : Tensor
            The input data, which is a tensor of shape (B, 3, H, W).
        """
        # if images.dtype == torch.uint8:
        #     images = images.float() / 255.0
        images = self.normalize(images)
        features = self.forward_extract(images)
        features = self.permute(features)

        return features

    def forward_extract(self, images: Tensor) -> BackboneResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} is missing required implemention!"
        )

    @classmethod
    def list_nodes(cls, name: str) -> tuple[list[str], list[str]]:
        msg = f"{cls.__name__} does not support node listing!"
        raise NotImplementedError(msg)

    @classmethod
    def list_models(
        cls, query: str | None = None, pretrained: bool = False, **kwargs
    ) -> list[str]:
        msg = f"{cls.__name__} does not support listing available models!"
        raise NotImplementedError(msg)

    def extra_repr(self) -> str:
        return f"model_name={self.model_name}, dimension_order={self.dimension_order}"


def hwc_to_chw(t: Tensor) -> Tensor:
    return t.permute(0, 3, 1, 2).contiguous()


######################################################
# Infer feature attributes applying random test data #
######################################################


_DEFAULT_TEST_SHAPE: T.Final = (512, 256)
_ReturnsFeatureDict: T.TypeAlias = T.Callable[[Tensor], dict[str, Tensor]]


def infer_shapes(
    mod: _ReturnsFeatureDict,
    test_shape: tuple[int, int] = _DEFAULT_TEST_SHAPE,
) -> dict[str, torch.Size]:
    r"""
    Infer the shapes of the features from the model by priming it with random data.

    Parameters
    ----------
    mod
        The model that returns a dict of features.
    test_shape
        The shape of the test data, which is used to infer the stride of the features.

    Returns
    -------
    dict[str, torch.Size]
        A dictionary mapping the feature names to their respective shapes.
    """

    mod_clone = copy.deepcopy(mod)
    try:
        mod_clone.eval()  # type: ignore
    except AttributeError:
        # logger.warning(R"Could not set %s to inference mode with `.eval()`", type(mod))
        pass
    with torch.no_grad():
        inp = torch.randn((8, 3, *test_shape), dtype=torch.float32)
        out = mod_clone(inp)
    if out is None or not out:
        msg = f"Failed to infer shapes from {type(mod)} with {test_shape=}. Invalid result {out=!r}"
        raise ValueError(msg)

    if not all(isinstance(v, Tensor) for v in out.values()):
        msg = f"Failed to infer shapes from {type(mod)} with {test_shape=}. Not all results are Tensors: {out}"
        raise ValueError(msg)

    shapes = {k: v.shape for k, v in out.items()}

    logger.debug(
        "Backbone shapes found using input shape %s to be %s",
        repr(test_shape),
        str(shapes),
    )

    return shapes


def infer_feature_info(
    mod: _ReturnsFeatureDict,
    dims: DimensionOrderType,
    test_shape: tuple[int, int] = _DEFAULT_TEST_SHAPE,
) -> BackboneFeatures:
    r"""
    Infer the feature information from the model by priming it with random data.

    Parameters
    ----------
    mod
        The model that returns a dict of features.
    dims
        The dimension order of the extracted features.
    test_shape
        The shape of the test data, which is used to infer the stride of the features.

    Returns
    -------
    dict[str, BackboneFeatureInfo]
        A dictionary mapping the feature names to their respective feature information.
    """
    shapes = infer_shapes(mod, test_shape)
    infos = {k: _shape_to_info(v, dims, test_shape) for k, v in shapes.items()}

    logger.debug("Backbone info found:\n%s", logger.create_table(infos))

    return infos


def _shape_to_info(
    shape: torch.Size,
    dims: DimensionOrderType,
    test_shape: tuple[int, int] = _DEFAULT_TEST_SHAPE,
) -> BackboneFeatureInfo:
    match dims:
        case DimensionOrder.CHW:
            c, h, w = map(int, shape[1:])
        case DimensionOrder.HWC:
            h, w, c = map(int, shape[1:])
        case _:
            raise NotImplementedError(f"Cannot infer feature info for {dims}")

    stride_h = test_shape[0] // h
    stride_w = test_shape[1] // w

    # TODO: This is not always true, e.g. for Swin Transformers with patch size 4 and stride 2, add support.
    assert stride_h == stride_w, (
        f"Stride must be equal in both dimensions, got {stride_h} and {stride_w}. "
        f"Size of input was {test_shape}, size of output was {shape}, order is {dims}."
    )

    stride = stride_h

    assert c > 0, c
    assert stride > 0, shape

    return BackboneFeatureInfo(channels=c, stride=stride)


# ------------------- #
# Catalog registry #
# ------------------- #


catalog: CatalogFromPackageMetadata[WrapperBase, T.Mapping[str, str]] = (
    CatalogFromPackageMetadata(require_info=False, group="unipercept.backbones")
)
