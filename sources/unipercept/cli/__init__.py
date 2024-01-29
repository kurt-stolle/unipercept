"""
This module contains the CLI commands for the `unipercept` package.
"""

from __future__ import annotations

import typing as T

import typing_extensions as TX

from .._monkeypatch import *
from . import backbones, echo, profile, trace, train
from ._command import *
from ._config import *
