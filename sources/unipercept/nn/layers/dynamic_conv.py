"""
Dynamic Convolution modules.
"""

from __future__ import annotations

import typing as T

import torch
import typing_extensions as TX
from einops import einsum, rearrange
from torch import Tensor, nn

from unipercept.nn.layers.activation import ActivationSpec, InplaceReLU
from unipercept.nn.layers.mlp import MLP
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
        d_f: int,
        d_k: int,
        d_h: int,
        layers: int | T.Tuple[int, int] | T.Sequence[int] = (1, 3),
        activation: ActivationSpec = InplaceReLU,
        dropout: float = 0.0,
        init_gain: float | T.Tuple[float, float] | T.Sequence[float] = (0.33, 0.33),
    ):
        r"""
        Parameters
        ----------
        d_f : int
            The number of dimensions in the feature tensor.
        d_k : int
            The number of dimensions in the kernel tensor.
        d_h : int
            The number of hidden dimensions.
        activation: ActivationSpec
            The activation function to use in projection layers.
        dropout : float
            The dropout rate in the projection layers.
        init_gain : float | Tuple[float,float] | Sequence[float]
            The gain to apply to the output tensor initialization.
        """

        super().__init__()

        self.d_f = d_f
        self.d_k = d_k
        self.d_o = d_h

        f_layers, k_layers = to_2tuple(layers)
        f_gain, k_gain = to_2tuple(init_gain)

        self.norm_k = nn.LayerNorm(d_k, elementwise_affine=False)
        self.norm_f = nn.LayerNorm(d_f, elementwise_affine=False)

        if k_layers > 0:
            self.proj_k = MLP(
                d_k,
                d_h,
                bias=None,
                layers=k_layers,
                activation=activation,
                dropout=dropout,
                init_gain=k_gain,
            )
        else:
            assert d_k == d_h, "If no kernel layers are used, d_k must be equal to d_o."
            self.proj_k = nn.Identity()

        if f_layers > 0:
            self.proj_f = MLP(
                d_f,
                d_h,
                bias=None,
                layers=f_layers,
                activation=activation,
                dropout=dropout,
                init_gain=f_gain,
            )
        else:
            assert (
                d_f == d_h
            ), "If no feature layers are used, d_f must be equal to d_o."
            self.proj_f = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the dynamic convolution.
        """
        for m in (self.proj_k, self.proj_f):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    @TX.override
    def forward(self, f: Tensor, k: Tensor) -> Tensor:
        r"""
        Perform dynamic convolution on the feature tensor.

        Parameters
        ----------
        f : Tensor[N, C_f, H, W]
            The feature tensor to convolve.
        k : Tensor[N, K, C_k]
            The kernel tensor to convolve with.

        Returns
        -------
        Tensor[N, K, H, W]
            The convolved tensor.
        """

        h, w = f.shape[-2:]
        f = rearrange(f, "... c h w -> ... (h w) c", h=h, w=w)
        f = self.proj_f(self.norm_f(f))
        k = self.proj_k(self.norm_k(k))

        o = einsum(k, f, "... n c, ... hw c -> ... n hw")
        o = rearrange(o, "... n (h w) -> ... n h w", h=h, w=w)

        return o


def dynamic_conv2d(
    features: torch.Tensor,
    kernels: torch.Tensor,
) -> torch.Tensor:
    """
    Perform dynamic convolution on the features using the flattened kernels.
    """
    return torch.einsum("... k c, ... c h w -> ... k h w", kernels, features)
