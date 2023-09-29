"""Implements the baseclass for a backbone, which is a feature extractor that returns a dict of features."""

from __future__ import annotations

import abc
import enum
import typing as T

import torch
from tensordict import TensorDict, TensorDictBase
from typing_extensions import override

from ._base import Backbone, BackboneFeatureInfo

__all__ = ["WrapperBase", "DimensionOrder", "ORDER_CHW", "ORDER_HWC", "infer_feature_info"]


class DimensionOrder(enum.StrEnum):
    """The format of the extracted features."""

    CHW = enum.auto()
    HWC = enum.auto()


ORDER_CHW = DimensionOrder.CHW
ORDER_HWC = DimensionOrder.HWC


class WrapperBase(Backbone, metaclass=abc.ABCMeta):
    """Baseclass for a backbone, which is a feature extractor that returns a dict of features."""

    dimension_order: T.Final[DimensionOrder]

    def __init__(
        self,
        *,
        dimension_order: str | DimensionOrder,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dimension_order = DimensionOrder(dimension_order)

    @override
    def forward(self, images: torch.Tensor) -> TensorDictBase:
        """
        Extract features from the input data.

        Parameters
        ----------
        images : torch.Tensor
            The input data, which is a tensor of shape (B, 3, H, W).
        """
        assert images.ndim == 4 and images.shape[1] == 3, f"Input must be of shape (B, 3, H, W), got {images.shape}"

        features = self.forward_extract(images)
        features = self.forward_permute(features)

        return features

    @abc.abstractmethod
    def forward_extract(self, images: torch.Tensor) -> TensorDictBase:
        raise NotImplementedError(f"{self.__class__.__name__} is missing required implemention!")

    def forward_permute(self, features: TensorDictBase) -> TensorDictBase:
        if self.dimension_order == DimensionOrder.CHW:
            return features
        if self.dimension_order == DimensionOrder.HWC:
            return features.apply(hwc_to_chw, batch_size=features.batch_size, inplace=not self.training)
        raise NotImplementedError(f"Cannot permute {self.dimension_order}")


def hwc_to_chw(t: torch.Tensor) -> torch.Tensor:
    return t.permute(0, 3, 1, 2)


def infer_shapes(
    mod: T.Callable[[torch.Tensor], T.List[torch.Tensor]], /, test_shape: tuple[int, int] = (512, 256)
) -> list[torch.Size]:
    inp = torch.randn(1, 3, *test_shape)
    with torch.no_grad():
        out = mod(inp)
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
    shapes = infer_shapes(mod, test_shape=test_shape)

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
