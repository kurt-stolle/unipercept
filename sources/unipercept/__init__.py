"""
UniPercept
----------

A framework for computer vision research and development based on PyTorch.
"""
from __future__ import annotations

__version__ = "5.1.1"

from . import config, data, engine, evaluators, log, model, nn, render, state, utils
from ._api_config import *
from ._api_data import *
from ._monkeypatch import *
