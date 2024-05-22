"""
Interface for the extension module.

See Also
--------

- ``extension.h`` : reference header file.

"""

from __future__ import annotations

from torch import Tensor

def deform_conv_backward(*args: Tensor) -> Tensor: ...
def deform_conv_forward(*args: Tensor) -> Tensor: ...
