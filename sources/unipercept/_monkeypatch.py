"""
Used to apply monkeypatching when the program is started.
"""

import torch

############################
# TENSOR SHAPE IN DEBUGGER #
############################


def __tensor_repr_patched(self: torch.Tensor) -> str:
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
    kwargs = ", ".join([f"{k}={getattr(self, k)}" for k in ["device", "requires_grad"]])

    # if self.ndim == 0:
    #     values = f"\t{self.item()}"
    # else:
    #     values = "\n".join([f"\t{v}" for v in self.tolist()])

    return f"{name}<{shape}>({dtype}, {kwargs})"  # {{\n{values}\n}}"


torch.Tensor.__repr__ = __tensor_repr_patched

#######################
