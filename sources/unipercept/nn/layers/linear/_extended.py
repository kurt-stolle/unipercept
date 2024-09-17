import typing as T

from torch import nn

from unipercept.nn.init import InitMixin
from unipercept.types import Tensor

__all__ = ["Linear"]


class Linear(InitMixin, nn.Linear):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @T.override
    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.linear(input, self.weight, self.bias)
