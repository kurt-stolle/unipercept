from __future__ import annotations

import torch
from torch import Tensor
from unipercept.utils.tensorclass import Tensorclass


def test_tensorclass_via_parent():
    """
    Our implementation should have the same signature as the decorator version
    """

    from tensordict import tensorclass

    class Foo(Tensorclass):
        a: Tensor
        b: Tensor
        non_tensor: str

    foo = Foo(Tensor(1), Tensor([1, 2, 3]), "baz", batch_size=[])

    @tensorclass
    class Bar:
        a: Tensor
        b: Tensor
        non_tensor: str

    a = torch.randn(1, 3, 3)
    b = torch.randn(1)
    non_tensor = "baz"
    batch_size = [1]

    # Test that the same operations work on both
    for cls in (Foo, Bar):
        # Initialize kwargs
        ins = cls(a=a, b=b, non_tensor=non_tensor, batch_size=batch_size)  # type: ignore

        # Initialize positional
        ins = cls(a, b, non_tensor, batch_size=batch_size)  # type: ignore
