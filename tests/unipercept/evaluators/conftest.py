r"""
Pytest configuration for ``unipercept.evaluators`` tests.
"""

from __future__ import annotations

import typing as T
from pathlib import Path

import pytest
from torch import Tensor

ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets" / "testing"
