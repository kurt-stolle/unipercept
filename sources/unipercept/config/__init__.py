"""
Lazy configuration system, inspired by and based on Detectron2 and Hydra.
"""

from __future__ import annotations

from . import env, lazy


def __getattr__(name: str):
    if name == "language":
        return ImportError("'unipercept.config.language' must be imported directly")
    msg = f"Module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
