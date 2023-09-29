from typing import Callable, Optional

import torch
from torch import Tensor, nn

__all__ = []

EPS_FLOAT = torch.finfo(torch.float).eps
EPS_HALF = torch.finfo(torch.half).eps
EPS_BF16 = torch.finfo(torch.bfloat16).eps


class NumStableLoss(nn.Module):
    def __init__(self, *args, eps: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)

        self.eps = eps

    def _nsb(self, x: Tensor, is_small: bool = False) -> Tensor:
        """
        Return a stable version of the input tensor.

        Parameters
        ----------
        x
            Input tensor
        is_small, optional
            If True, clamp the input tensor to a small value, otherwise add a small value.
            The latter is preferred when the input tensor is (generally) close to zero.

        Returns
        -------
            Numerically stable version of the input tensor.
        """

        if is_small:
            return x + self.eps
        else:
            return x.clamp(min=self.eps)


class ReducableLoss(nn.Module):
    def __init__(
        self,
        *args,
        reduction: str | Callable[[Tensor], Tensor] = "none",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.reduction = reduction

    def _reduce(self, loss: Tensor) -> Tensor:
        """
        Reduce the loss tensor.
        """
        rt = self.reduction

        # Reduction switch by name
        if isinstance(rt, str):
            if rt == "sum":
                return loss.sum()
            elif rt == "mean":
                return loss.mean()
            elif rt == "none":
                return loss

        # Callable reduction type
        if callable(rt):
            return rt(loss)

        # Reduction is not supported
        raise NotImplementedError(f"Reduction {rt} ({type(rt)}) is not supported!")
