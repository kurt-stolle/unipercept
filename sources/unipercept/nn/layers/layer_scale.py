r"""
Feature scaling layers.
"""

import typing_extensions as TX
import torch
import torch.nn as nn

from unipercept.types import Tensor


class LayerScale(nn.Module):
    r"""
    Implementation of the LayerScale methodology used in ConvNextV2.
    """

    def __init__(
        self,
        dim: int,
        gamma: float | Tensor = 1e-5,
        inplace: bool | None = False,
    ) -> None:
        r"""
        Parameters
        ----------
        dim : int
            The number of features in the input tensor.
        gamma : float or Tensor
            The initial value of the scaling factor.
        inplace : bool, optional
            Whether to perform the operation in-place or not. If `None`, the operation
            is performed in-place if the input tensor is contiguous and the
            module is *not* in training mode.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(gamma * torch.ones(dim))

    @TX.override
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace is True or (
            self.inplace is None and not self.training and x.is_contiguous()
        ):
            return x.mul_(self.gamma)
        return x * self.gamma
