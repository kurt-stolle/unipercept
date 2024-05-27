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
        **kwargs,
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
        **kwargs
            Keyword arguments for the projection layers.
        """

        super().__init__()

        self.kernel_dim = kernel_dim
        self.feature_dim = feature_dim
        self.conv_dim = conv_dim

        self.proj_kernel = MLP(kernel_dim, conv_dim, **kwargs)
        self.proj_feature = MLP(feature_dim, conv_dim, **kwargs)

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

        k = self.proj_kernel(kernel)
        f = rearrange(feature, "... c h w -> ... (h w) c")
        f = self.proj_feature(f)
        o = einsum(k, f, "... n c, ... (h w) c -> ... n (h w)")
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
