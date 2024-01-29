"""
Various shorthand type definitions, to avoid code repetition.
"""

from __future__ import annotations

import pathlib
import datetime
import os

__all__ = ["Pathable", "Buffer"]

Buffer = bytes | bytearray | memoryview

Pathable = str | pathlib.Path | os.PathLike

Primitive = int | float | str | bytes | bytearray | memoryview

Datetime = datetime.datetime
