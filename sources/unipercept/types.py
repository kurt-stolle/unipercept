r"""
Defines commin typings used throughout all submodules.
"""

import datetime
import os
import pathlib
import typing as T

import torch
import torch.nn
import torch.types

Tensor: T.TypeAlias = torch.Tensor
Device: T.TypeAlias = torch.device | torch.types.Device
DType: T.TypeAlias = torch.dtype
StateDict: T.TypeAlias = dict[str, Tensor]
Size: T.TypeAlias = torch.Size | tuple[int, ...]
Buffer = bytes | bytearray | memoryview
Pathable = str | pathlib.Path | os.PathLike
Primitive = int | float | str | bytes | bytearray | memoryview
Datetime = datetime.datetime
