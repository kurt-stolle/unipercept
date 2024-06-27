r"""
Defines commin typings used throughout all submodules.

Some types are only defined as a stub, where the full definition is in the interface 
file (``types.pyi``).
"""

from __future__ import annotations

import typing as T


import torch
import torch.types
import torch.nn

Tensor: T.TypeAlias = torch.Tensor
Device: T.TypeAlias = torch.device | torch.types.Device
DType: T.TypeAlias = torch.dtype
StateDict: T.TypeAlias = T.Dict[str, Tensor]
