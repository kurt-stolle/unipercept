import typing as T

import torch
from torch import nn

from unipercept.nn.activations import ActivationSpec, SiLU, get_activation
from unipercept.nn.norms import LayerNorm, NormSpec, get_norm
from unipercept.types import Tensor

__all__ = ["RadialLinear", "MRP"]

_DEFAULT_ACTIVATION = SiLU
_DEFAULT_NORM = LayerNorm


class _Spline(nn.Linear):
    r"""
    Handles initialization of the spline weights.
    """

    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw
    ) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    @T.override
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

    if T.TYPE_CHECKING:
        __call__ = nn.Linear.forward


class _RBF(nn.Module):
    """
    Defines a radial basis function layer
    """

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float | None = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    @T.override
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))

    if T.TYPE_CHECKING:
        __call__ = forward


class RadialLinear(nn.Module):
    """
    Experimental approximate Linear layer using radial basis functions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        base_update: bool = True,
        base_activation: ActivationSpec = _DEFAULT_ACTIVATION,
        spline_weight_init_scale: float = 0.1,
        norm: NormSpec = _DEFAULT_NORM,
    ) -> None:
        super().__init__()
        self.norm = get_norm(norm, in_features)
        self.rbf = _RBF(grid_min, grid_max, num_grids)
        self.spline_linear = _Spline(
            in_features * num_grids, out_features, spline_weight_init_scale
        )
        if base_update:
            self.base_activation = get_activation(base_activation)
            self.base_linear = nn.Linear(in_features, out_features)
        else:
            self.register_module("base_activation", None)
            self.register_module("base_linear", None)

    @T.override
    def forward(self, x: Tensor) -> Tensor:
        spline_basis = self.rbf(self.norm(x))
        y = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.base_linear is not None:
            y = y + self.base_linear(self.base_activation(x))
        return y

    if T.TYPE_CHECKING:
        __call__ = forward


class MRP(nn.Module):
    """
    Multi-layer radial perceptron
    """

    def __init__(
        self,
        layers: T.Sequence[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        base_update: bool = True,
        base_activation=nn.functional.silu,
        spline_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RadialLinear(
                    in_dim,
                    out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    base_update=base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_init_scale,
                )
                for in_dim, out_dim in zip(layers[:-1], layers[1:], strict=False)
            ]
        )

    @T.override
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    if T.TYPE_CHECKING:
        __call__ = forward
