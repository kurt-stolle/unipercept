from __future__ import annotations

import enum as E
import typing as T

import torch
import torch.fx
import typing_extensions as TX
from torch import Tensor, nn


class DropPathMode(E.StrEnum):
    """
    The drop mode for :func:`drop_path`.
    """

    BATCH = "batch"
    ROW = "row"


class DropPath(nn.Module):
    """
    Module that wraps :func:`drop_path`.
    """

    def __init__(
        self,
        p: float,
        mode: DropPathMode | T.Literal["row", "batch"] = DropPathMode.ROW,
    ) -> None:
        """
        Parameters
        ----------
        p : float
            The drop probability.
        mode : DropPathMode, optional
            The drop mode, by default DropPathMode.ROW. See :func:`drop_path` for details.
        """
        super().__init__()
        self.p = p
        self.mode = DropPathMode(mode)

    @TX.override
    def forward(self, input: Tensor) -> Tensor:
        """
        Applies drop path to the input tensor.

        Parameters
        ----------
        input : Tensor[*shape]
            The input tensor.

        Returns
        -------
        Tensor[*shape]
            The output tensor.
        """
        return drop_path(input, self.p, self.training, self.mode)

    @TX.override
    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


def drop_path(
    input: Tensor, p: float, training: bool, mode: DropPathMode | str = DropPathMode.ROW
) -> Tensor:
    """
    Implements stochastic depth from [1]

    Parameters
    ----------
    input : Tensor[*shape]
        The input tensor.
    p : float
        The drop probability.
    mode : DropPathMode
        The drop mode, either "batch" or "row".
        In "batch" mode, the same mask is applied to all elements in the batch.
        In "row" mode, a different mask is applied to each row in the batch.
    training : bool, optional
        Whether the model is in training mode, by default True.

    Returns
    -------
    Tensor[*shape]
        The output tensor.

    References
    ----------
    [1] `"Deep Networks with Stochastic Depth" <https://arxiv.org/abs/1603.09382>`_.
    """
    if not training or p == 0.0:
        return input

    match mode:
        case DropPathMode.BATCH:
            size = [1] * input.ndim
        case DropPathMode.ROW:
            size = [input.shape[0]] + [1] * (input.ndim - 1)
        case _:
            msg = f"Drop path mode not implemented: {mode}"
            raise NotImplementedError(msg)

    survival_rate = 1.0 - p
    noise = input.new_empty(size).bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


torch.fx.wrap("drop_path")
