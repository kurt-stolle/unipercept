"""
Used to apply monkeypatching when the program is started.
"""

from __future__ import annotations

import typing as T

__all__ = []
__dir__ = []

############################
# TENSOR SHAPE IN DEBUGGER #
############################


def __patch_tensor_repr():
    """
    By default, `torch.Tensor.__repr__` shows the value of the tensor. This patch changes that to show the shape of the
    tensor instead. This is useful when debugging, as it allows to quickly see the shape of a tensor without having to
    print it, and also speeds up the debugger by not having to print large tensors.
    """

    import torch

    def patch(self: torch.Tensor, *args, **kwargs) -> str:
        """
        Patched version of `torch.Tensor.__repr__` that shows the shape of the tensor, instead of its value (which is the
        default behavior).
        """
        if self.ndim == 0:
            shape = "*"
        else:
            shape = str(tuple(self.shape)).replace(" ", "").replace(",", "×")[1:-1]
        name = self.__class__.__name__
        dtype = str(self.dtype).replace("torch.", "")
        kwargs = ", ".join(
            [f"{k}={getattr(self, k)}" for k in ["device", "requires_grad"]]
        )

        # if self.ndim == 0:
        #     values = f"\t{self.item()}"
        # else:
        #     values = "\n".join([f"\t{v}" for v in self.tolist()])

        return f"{name}<{shape}>({dtype}, {kwargs})"  # {{\n{values}\n}}"

    torch.Tensor.__repr__ = patch


__patch_tensor_repr()

##########################
# HUGGINGFACE ACCELERATE #
##########################


def _update_references(old, new):
    import gc

    gc.collect()

    for ref in gc.get_referrers(old):
        try:
            if "__name__" not in ref:
                continue
            for key in list(ref.keys()):
                if ref[key] is old:
                    ref[key] = new
        except Exception:
            pass

    gc.collect()


def __patch_accelerate_operations():
    """
    TensorDict and Accelerate do not play well together. This patch fixes that.
    """
    import accelerate.utils.operations
    import torch

    def is_tensordict_like(data: T.Any) -> bool:
        from tensordict import TensorDictBase

        from unipercept.utils.tensorclass import Tensorclass

        return isinstance(data, (TensorDictBase, Tensorclass))

    # PATCH 1: `accelerate.utils.operations.recursively_apply`
    recursively_apply__original = accelerate.utils.operations.recursively_apply

    def recursively_apply(
        fn: T.Callable[[T.Any], T.Any],
        data: T.Any,
        *args: T.Any,
        **kwargs: T.Any,
    ) -> T.Any:
        """
        Patched version of `accelerate.utils.operations.recursively_apply` that supports `TensorDict`.
        """
        if is_tensordict_like(data):
            return data.apply(fn)
        else:
            return recursively_apply__original(fn, data, *args, **kwargs)

    accelerate.utils.operations.recursively_apply = recursively_apply
    _update_references(recursively_apply__original, recursively_apply)

    # PATCH 2: `accelerate.utils.operations.concatenate`
    concatenate__original = accelerate.utils.operations.concatenate

    def concatenate(
        data: T.Sequence[T.Any],
        dim: int = 0,
    ) -> T.Any:
        """
        Patched version of `accelerate.utils.operations.concatenate` that supports `TensorDict`.
        """
        if is_tensordict_like(data[0]):
            return torch.cat(data, dim=dim)  # type: ignore
        else:
            return concatenate__original(data, dim=dim)

    accelerate.utils.operations.concatenate = concatenate

    _update_references(concatenate__original, concatenate)


__patch_accelerate_operations()
