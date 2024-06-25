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
from unipercept.nn.layers.norm import NormSpec, get_norm
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
        layers: int | T.Tuple[int, int] | T.Iterable[int] = (1, 3),
        layer_expansion: (
            float
            | T.Tuple[float | None, float | None]
            | T.Sequence[float | None]
            | None
        ) = 2.0,
        activation: ActivationSpec = InplaceReLU,
        dropout: float = 0.0,
        input_norm: (
            NormSpec
            | T.Tuple[NormSpec | None, NormSpec | None]
            | T.Sequence[NormSpec | None]
            | None
        ) = None,
        init_gain: (
            float
            | T.Tuple[float | None, float | None]
            | T.Sequence[float | None]
            | None
        ) = (0.33, 0.33),
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
        layers : int | Tuple[int,int] | Sequence[int]
            The number of layers in the projection MLPs for the feature and kernel,
            respectively. If a single integer is provided, the same number of layers
            is used for both projections.
        layer_expansion : float | Tuple[float,float] | Sequence[float] | None
            The expansion factor to use in the projection layers. If a single float is
            provided, the same factor is used for both projections. When no layers
            are used, the expansion factor must be explicitly set to None.
        activation: ActivationSpec
            The activation function to use in projection layers.
        dropout : float
            The dropout rate in the projection layers.
        input_norm : NormSpec | Tuple[NormSpec,NormSpec] | Sequence[None]
            The normalization layer to apply to the input tensors. If a single norm
            is provided, the same norm is used for both inputs. If None, no normalization
            is applied.
        init_gain : float | Tuple[float,float] | Sequence[float]
            The gain to apply to the output tensor initialization. When no layers
            are used, the expansion factor must be explicitly set to None.
        """

        super().__init__()

        self.d_f = d_f
        self.d_k = d_k
        self.d_o = d_h

        f_layers, k_layers = to_2tuple(layers)
        f_gain, k_gain = to_2tuple(init_gain)
        f_exp, k_exp = to_2tuple(layer_expansion)
        f_norm, k_norm = to_2tuple(input_norm)
        if f_norm is not None:
            self.norm_f = get_norm(f_norm, d_f)
        else:
            self.register_module("norm_f", f_norm)

        if k_norm is not None:
            self.norm_k = get_norm(k_norm, d_k)
        else:
            self.register_module("norm_k", k_norm)

        # Kernel projection
        if k_layers > 0:
            assert k_exp is not None
            self.proj_k = MLP(
                d_k,
                d_h,
                hidden_features=k_exp,
                bias=None,
                layers=k_layers,
                activation=activation,
                dropout=dropout,
                init_gain=k_gain,
            )
        else:
            assert (
                d_k == d_h
            ), f"If no kernel layers are used, {d_k=} must be equal to {d_h=}."
            assert (
                k_exp is None
            ), "If no kernel layers are used, expansion factor must be None."
            assert (
                k_gain is None
            ), "If no kernel layers are used, init gain must be None."
            self.proj_k = nn.Identity()

        # Feature projection
        if f_layers > 0:
            assert f_exp is not None
            self.proj_f = MLP(
                d_f,
                d_h,
                hidden_features=f_exp,
                bias=None,
                layers=f_layers,
                activation=activation,
                dropout=dropout,
                init_gain=f_gain,
            )
        else:
            assert (
                d_f == d_h
            ), f"If no feature layers are used, {d_f=} must be equal to {d_h=}."
            assert (
                f_exp is None
            ), "If no feature layers are used, expansion factor must be None."
            assert (
                f_gain is None
            ), "If no feature layers are used, init gain must be None."
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

        # Feature projection
        f = rearrange(f, "... c h w -> ... (h w) c", h=h, w=w)
        if self.norm_f is not None:
            f = self.norm_f(f)
        f = self.proj_f(f)

        # Kernel projection
        if self.norm_k is not None:
            k = self.norm_k(k)
        k = self.proj_k(k)

        # Dynamic convolution
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
