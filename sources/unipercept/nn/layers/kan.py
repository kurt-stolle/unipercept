r"""
Implementation of KAN (Kolmogorov-Arnold Network) layers [1].

Notes
-----
This implementation is a work in progress. GitHub Copilot was used to generate the
initial structure and some implementation from the original paper.

Mission critical applications may want to use a more mature implementation, such as 
`PyKAN <https://github.com/kindxiaoming/pykan>`_.

References
----------

[1] `KAN: Kolmogorow-Arnold Networks <https://arxiv.org/pdf/2404.19756v2>`_
"""

from __future__ import annotations

import math
import typing as T

import torch
import typing_extensions as TX
from torch import Tensor, nn

from unipercept.nn.layers.activation import ActivationSpec, get_activation


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        standalone_scale_spline=True,
        base_activation: ActivationSpec = nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            Tensor(out_features, in_features, grid_size + spline_order)
        )
        if standalone_scale_spline:
            self.spline_scaler = nn.Parameter(Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.standalone_scale_spline = standalone_scale_spline
        self.activation = get_activation(base_activation)
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.standalone_scale_spline else 1.0)
                * self.curve_to_coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.standalone_scale_spline:
                # nn.init.constant_(self.spline_scaler, self.scale_spline)
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: Tensor) -> Tensor:
        """
        Compute the B-spline bases for the given input tensor.

        Parameters
        ----------
        x : Tnesor[B,F]
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        Tensor[B,F,G*S]
            B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: Tensor = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve_to_coeff(self, x: Tensor, y: Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Parameters
        ----------
        x : Tensor[B, I]
            Input tensor of shape (batch_size, in_features).
        y : Tensor[B, I, O]
            Output tensor of shape (batch_size, in_features, out_features).

        Returns
        -------
        Tensor[O,I,G*S]
            Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.standalone_scale_spline else 1.0
        )

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = nn.functional.linear(self.activation(x), self.base_weight)
        spline_output = nn.functional.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve_to_coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )

    if T.TYPE_CHECKING:
        __call__ = forward


class KAN(nn.Module):
    def __init__(
        self,
        layers,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation: ActivationSpec = nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList(
            [
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
                for in_features, out_features in zip(layers[:-1], layers[1:])
            ]
        )

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    if T.TYPE_CHECKING:
        __call__ = forward


class _Spline(nn.Linear):
    r"""
    Handles initialization of the spline weights.
    """

    def __init__(
        self, in_features: int, out_features: int, init_scale: float = 0.1, **kw
    ) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    @TX.override
    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

    if T.TYPE_CHECKING:
        __call__ = nn.Linear.forward


class _RBF(nn.Module):
    """
    Defines a radial basis function layer for KAN approximation
    """

    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(-(((x[..., None] - self.grid) / self.denominator) ** 2))

    if T.TYPE_CHECKING:
        __call__ = forward


class RadialKANLinear(nn.Module):
    """
    Experimental approximate KANLinear layer using radial basis functions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        base_update: bool = True,
        base_activation: ActivationSpec = nn.SiLU,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(in_features)
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

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        spline_basis = self.rbf(self.layernorm(x))
        y = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.base_linear is not None:
            y = y + self.base_linear(self.base_activation(x))
        return y

    if T.TYPE_CHECKING:
        __call__ = forward


class RadialKAN(nn.Module):
    """
    Experimental approximate KAN layer using radial basis functions.
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
                RadialKANLinear(
                    in_dim,
                    out_dim,
                    grid_min=grid_min,
                    grid_max=grid_max,
                    num_grids=num_grids,
                    base_update=base_update,
                    base_activation=base_activation,
                    spline_weight_init_scale=spline_init_scale,
                )
                for in_dim, out_dim in zip(layers[:-1], layers[1:])
            ]
        )

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    if T.TYPE_CHECKING:
        __call__ = forward
