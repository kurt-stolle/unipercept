"""Implements the baseclass for a backbone, which is a feature extractor that returns a dict of features."""

from __future__ import annotations

import abc
import enum
import typing as T

import torch
from tensordict import TensorDictBase
from typing_extensions import override

from ._base import Backbone, BackboneFeatureInfo
from ._normalize import Normalizer

__all__ = [
    "WrapperBase",
    "DimensionOrder",
    "ORDER_CHW",
    "ORDER_HWC",
    "infer_feature_info",
]


class DimensionOrder(enum.Enum):
    """The format of the extracted features."""

    CHW = enum.auto()
    HWC = enum.auto()


ORDER_CHW = DimensionOrder.CHW
ORDER_HWC = DimensionOrder.HWC


class WrapperBase(Backbone, metaclass=abc.ABCMeta):
    """
    Baseclass for a backbone, which is a feature extractor that returns a dict of features.

    Parameters
    ----------
    dimension_order : str | DimensionOrder
        The format of the **extracted features**. Note that this is not the format of the input data, which is always
        assumed to be (B, 3, H, W) in RGB format!
    mean : list[float]
        The mean values for each channel, which are used for normalization.
    std : list[float]
        The standard deviation values for each channel, which are used for normalization.
    """

    dimension_order: T.Final[DimensionOrder]

    def __init__(
        self,
        *,
        dimension_order: str | DimensionOrder,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dimension_order = DimensionOrder(dimension_order)
        self.normalize = Normalizer(mean, std)

    @override
    def forward(self, images: torch.Tensor) -> TensorDictBase:
        """
        Extract features from the input data.

        Parameters
        ----------
        images : torch.Tensor
            The input data, which is a tensor of shape (B, 3, H, W).
        """
        # if images.dtype == torch.uint8:
        #     images = images.float() / 255.0
        images = self.normalize(images)

        features = self.forward_extract(images)
        features = self.forward_permute(features)

        return features

    @abc.abstractmethod
    def forward_extract(self, images: torch.Tensor) -> TensorDictBase:
        raise NotImplementedError(
            f"{self.__class__.__name__} is missing required implemention!"
        )

    def forward_permute(self, features: TensorDictBase) -> TensorDictBase:
        if self.dimension_order == DimensionOrder.CHW:
            return features
        if self.dimension_order == DimensionOrder.HWC:
            return features.apply(
                hwc_to_chw, batch_size=features.batch_size, inplace=not self.training
            )
        raise NotImplementedError(f"Cannot permute {self.dimension_order}")


def hwc_to_chw(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 3, 1, 2)


def infer_shapes(
    mod: T.Callable[[torch.Tensor], T.List[torch.Tensor]], test_shape: tuple[int, int]
) -> list[torch.Size]:
    import copy

    mod_clone = copy.deepcopy(mod)
    with torch.inference_mode():
        inp = torch.randn((8, 3, *test_shape), dtype=torch.float32, requires_grad=False)
        out = mod_clone(inp)
        if isinstance(out, dict):
            out = list(out.values())
        shapes = [o.shape for o in out]

    return shapes


def infer_feature_info(
    mod: T.Callable[[torch.Tensor], T.List[torch.Tensor]],
    dims: DimensionOrder | str,
    /,
    test_shape: tuple[int, int] = (512, 256),
) -> tuple[BackboneFeatureInfo, ...]:
    """Information about the extracted features."""
    shapes = infer_shapes(mod, test_shape)

    def __infer():
        for shape in shapes:
            match DimensionOrder(dims):
                case DimensionOrder.CHW:
                    c, h, w = shape[1:]
                case DimensionOrder.HWC:
                    h, w, c = shape[1:]
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

            yield BackboneFeatureInfo(channels=c, stride=stride)

    return tuple(__infer())
