"""
Dynamic Convolution modules.
"""

import typing as T

import torch
import typing_extensions as TX
from torch import nn

from unipercept.log import logger
from unipercept.nn._args import to_2tuple
from unipercept.nn.activations import ActivationSpec, InplaceReLU
from unipercept.nn.init import init_xavier_fill_
from unipercept.nn.layers.linear._mlp import MLP
from unipercept.nn.norms import NormSpec, get_norm
from unipercept.types import Tensor

########################
# Modules and wrappers #
########################


class DynamicConv2d(nn.Module):
    """
    Module that performs a dynamic convolution between a set of kernels and a feature map.

    The kernels are a set of K vectors, which are mapped and reshaped to K tensors that
    are convolved with the feature map of shape (... H W).

    The output is a tensor of shape (... K H W).
    """

    @TX.override
    def forward(self, features: Tensor, kernels: Tensor) -> Tensor:
        """
        Perform dynamic convolution on the features using the flattened kernels.
        """
        return dynamic_conv2d(features, kernels)


class ProjectConv2d(nn.Module):
    """
    Dynamic convolution over 2D feature map, with optional projection layers.
    """

    feature_norm: nn.Module | None
    feature_ffn: nn.Module | None

    kernel_norm: nn.Module | None
    kernel_ffn: nn.Module | None

    def __init__(
        self,
        d_feature: int,
        d_kernel: int,
        d_hidden: int,
        layers: int | tuple[int, int] | T.Iterable[int] = (1, 3),
        layer_expansion: (
            float | tuple[float | None, float | None] | T.Sequence[float | None] | None
        ) = 2.0,
        activation: ActivationSpec = InplaceReLU,
        dropout: float = 0.0,
        input_norm: (
            NormSpec
            | tuple[NormSpec | None, NormSpec | None]
            | T.Sequence[NormSpec | None]
            | None
        ) = None,
        init_gain: (
            float | tuple[float | None, float | None] | T.Sequence[float | None] | None
        ) = (0.33, 0.33),
        bias: tuple[bool | None, bool | None] | bool | None = None,
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

        self.d_f = d_feature
        self.d_k = d_kernel
        self.d_o = d_hidden

        # Generate layers for each branch
        for (
            br_name,
            br_dim,
            br_layers,
            br_gain,
            br_expansion,
            br_dropout,
            br_norm,
            br_activation,
            br_bias,
        ) in zip(
            ("feature", "kernel"),
            (d_feature, d_kernel),
            to_2tuple(layers),
            to_2tuple(init_gain),
            to_2tuple(layer_expansion),
            to_2tuple(dropout),
            to_2tuple(input_norm),
            to_2tuple(activation),
            to_2tuple(bias),
            strict=True,
        ):
            norm = ffn = None

            if br_norm is not None:
                norm = get_norm(br_norm, br_dim)
            if br_layers > 1:
                ffn = MLP(
                    br_dim,
                    d_hidden,
                    hidden_features=br_expansion,
                    bias=None,
                    layers=br_layers,
                    activation=br_activation,
                    dropout=br_dropout,
                    init_gain=br_gain,
                )
            else:
                if br_layers == 1:
                    ffn = nn.Linear(br_dim, d_hidden, bias=bool(br_bias))
                    init_xavier_fill_(ffn)
                    if br_gain is not None:
                        ffn.weight.data.normal_(mean=0.0, std=br_gain)
                elif br_dim != d_hidden:
                    msg = (
                        f"{br_name} layers must be non-negative to map input dimension "
                        f"({br_dim}) to hidden dimension ({d_hidden})."
                    )
                    raise ValueError(msg)
                elif br_gain is not None and br_gain != 1.0:
                    logger.warning(f"No layers used, {br_name} init gain ignored.")

                # Warn the user about unused arguments (should be None to avoid this)
                if br_expansion is not None and br_expansion != 1.0:
                    logger.warning(
                        f"Less than two layers used, {br_name} expansion ignored."
                    )
                if br_dropout > 0.0:
                    logger.warning(
                        f"Less than two layers used, {br_name} dropout ignored."
                    )
                if br_activation is not None:
                    logger.warning(
                        f"Less than two layers used, {br_name} activation ignored."
                    )

            self.register_module(f"{br_name}_norm", norm)
            self.register_module(f"{br_name}_ffn", ffn)

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

        _, _, h, w = f.shape

        # Feature projection
        # f = rearrange(f, "... c h w -> ... (h w) c", h=h, w=w)
        f = f.flatten(2)  # (B, C, H*W)
        f = f.transpose(1, 2)  # (B, H*W, C)

        if self.feature_norm is not None:
            f = self.feature_norm(f)
        if self.feature_ffn is not None:
            f = self.feature_ffn(f)

        # Kernel projection
        if self.kernel_norm is not None:
            k = self.kernel_norm(k)
        if self.kernel_ffn is not None:
            k = self.kernel_ffn(k)

        # Dynamic convolution
        # o = einsum(k, f, "... n c, ... hw c -> ... n hw")
        o = torch.einsum("bkc,bsc->bks", k, f)
        # o = rearrange(o, "... n (h w) -> ... n h w", h=h, w=w)
        o = o.unflatten(-1, (h, w))
        o = o.contiguous()

        return o


###################
# Implementations #
###################
def _dynamic_conv2d_naive(
    feature: Tensor,  # (B, C, H, W)
    kernels: Tensor,  # (B, K, C)
) -> Tensor:
    batch_size = feature.shape[0]

    feature = feature.flatten(0, 1)  # (B*C, H, W)
    feature = feature.unsqueeze(0)  # (1, B*C, H, W)

    kernels = kernels.unflatten(-1, (-1, 1, 1))  # (B, K, C, 1, 1)
    kernels = kernels.flatten(0, 1)  # (B*K, C, 1, 1)

    results = nn.functional.conv2d(
        feature, kernels, stride=1, padding=0, groups=batch_size
    )  # (1, B*K, H, W)
    results = results.squeeze(0).unflatten(0, (batch_size, -1))  # (B, K, H, W)
    return results.contiguous()


def _dynamic_conv2d_einsum(
    feature: Tensor,  # (B, C, H, W)
    kernels: Tensor,  # (B, K, C)
) -> Tensor:
    return torch.einsum("bkc, bchw -> bkhw", kernels, feature).contiguous()


def _dynamic_conv2d_matmul(
    feature: Tensor,  # (B, C, H, W)
    kernels: Tensor,  # (B, K, C)
) -> Tensor:
    # (K, C) x (C, H*W) -> (K, H*W)
    return (
        torch.bmm(kernels, feature.flatten(2))
        .unflatten(-1, feature.shape[2:])
        .contiguous()
    )


def dynamic_conv2d(
    feature: Tensor,
    kernels: Tensor,
) -> Tensor:
    """
    Perform dynamic convolution on the features using the flattened kernels.
    """

    # Use the fastest implementation available
    return _dynamic_conv2d_matmul(feature, kernels)
