"""
Implements a wrapper for models that are trained on the UniPercept dataset common format.
"""

from __future__ import annotations

import typing as T

import torch
import torch.nn as nn

__all__ = ["PerceptionModel"]

if T.TYPE_CHECKING:
    from unipercept.data.points import InputData

    from .typings import DictModule


class PerceptionModel(nn.Module):
    """
    Baseclass for models that are trained on the UniPercept dataset common format.
    """

    def __init__(
        self,
        model: DictModule,
        *,
        replace_batch_norm=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = T.cast(nn.Module, model)

        if replace_batch_norm:
            # HACK: The current setup does not support BatchNorm due to having added support for a time dimension
            # in the input. This can be solved using the provided utility in `torch.func`, but this solution
            # is noted to be not ideal in their documentation.
            from torch.func import replace_all_batch_norm_modules_

            replace_all_batch_norm_modules_(self.model)

    def forward(self, data: InputData) -> T.Dict[str, torch.Tensor]:
        """
        Runs the model on the given input data.
        """

        return self.model(data)
