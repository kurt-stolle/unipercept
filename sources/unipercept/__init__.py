"""

UniPercept: A framework for computer vision research based on PyTorch.

======================================================================

"""

from __future__ import annotations

__version__ = "5.1.6"

from unipercept._monkeypatch import *  # isort: skip; noqa: F401, F403; black: skip

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
    types,
    utils,
    vision,
)
from unipercept._api_config import *
from unipercept._api_data import *

__all__ = [
    "config",
    "data",
    "engine",
    "evaluators",
    "file_io",
    "log",
    "types",
    "vision",
    "model",
    "nn",
    "render",
    "state",
    "utils",
]
