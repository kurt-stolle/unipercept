"""
UniPercept
----------

A framework for computer vision research and development based on PyTorch.
"""
from __future__ import annotations

__version__ = "5.1.1"

from unipercept import (
    config,
    data,
    engine,
    evaluators,
    file_io,
    log,
    model,
    nn,
    render,
    state,
    utils,
)
from unipercept._api_config import *
from unipercept._api_data import *
from unipercept._monkeypatch import *
