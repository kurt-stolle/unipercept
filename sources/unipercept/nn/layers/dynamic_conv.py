"""
Dynamic Convolution modules.
"""

from __future__ import annotations
import typing as T
import typing_extensions as TX
import torch

from torch import nn, Tensor

from einops import rearrange, einsum

from unipercept.nn.layers.activation import ActivationSpec, InplaceReLU
from unipercept.nn.layers.mlp import MLP
from unipercept.nn.layers.conv import Conv2d
from unipercept.nn.layers.utils import to_2tuple


class DynamicConv2d(nn.Module):
    """
    Module that performs a dynamic convolution between a set of kernels and a feature map.

    The kernels are a set of K vectors, which are mapped and reshaped to K tensors that
    are convolved with the feature map of shape (... H W).

    The output is a tensor of shape (... K H W).
    """

    def __init__(
        self,
        feature_dim: int,
        kernel_dim: int,
        conv_dim: int,
        layers: int | T.Tuple[int,int] | T.Sequence[int] = (2,1),
        activation: ActivationSpec = InplaceReLU,
        dropout: float = 0.0,
        init_gain: float | T.Tuple[float,float] | T.Sequence[float] = (0.01,1.0)
    ):
        r"""
        Parameters
        ----------
        kernel_dim : int
            The number of dimensions in the kernel tensor.
        feature_dim : int
            The number of dimensions in the feature tensor.
        conv_dim : int
            The number of dimensions in the output tensor.
        activation: ActivationSpec
            The activation function to apply to the output tensor.
        dropout : float
            The dropout rate to apply to the output tensor.
        init_gain : float | Tuple[float,float] | Sequence[float]
            The gain to apply to the output tensor initialization.
        """

        super().__init__()

        self.kernel_dim = kernel_dim
        self.feature_dim = feature_dim
        self.conv_dim = conv_dim

        layers_kernel, layers_feature = to_2tuple(layers)
        gain_kernel, gain_feature = to_2tuple(init_gain)

        self.proj_kernel = MLP(
            kernel_dim,
            conv_dim,
            bias=None,
            layers=layers_kernel,
            activation=activation,
            dropout=dropout,
            init_gain=gain_kernel,
        )
        self.proj_feature = MLP(
            feature_dim,
            conv_dim,
            bias=None,
            layers=layers_feature,
            activation=activation,
            dropout=dropout,
            init_gain=gain_feature,
        )

    @TX.override
    def forward(self, feature: Tensor, kernel: Tensor) -> Tensor:
        r"""
        Perform dynamic convolution on the feature tensor.

        Parameters
        ----------
        feature : Tensor[N, C_f, H, W]
            The feature tensor to convolve.
        kernel : Tensor[N, K, C_k]
            The kernel tensor to convolve with.

        Returns
        -------
        Tensor[N, K, H, W]
            The convolved tensor.
        """

        f = rearrange(feature, "... c h w -> ... (h w) c")
        f = self.proj_feature(f)
        k = self.proj_kernel(kernel)
        o = einsum(k, f, "... n c, ... hw c -> ... n hw")
        o = rearrange(o, "... n (h w) -> ... n h w", w=feature.shape[-1])

        return o


def dynamic_conv2d(
    features: torch.Tensor,
    kernels: torch.Tensor,
) -> torch.Tensor:
    """
    Perform dynamic convolution on the features using the flattened kernels.
    """
    return torch.einsum("... k c, ... c h w -> ... k h w", kernels, features)
