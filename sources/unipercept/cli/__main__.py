"""
Entry point for the CLI.
"""

from __future__ import annotations

import sys

from unipercept.cli import (
    backbones,
    datasets,
    echo,
    evaluate,
    export,
    path,
    profile,
    train,
)

from ._command import command

__all__ = [
    "backbones",
    "echo",
    "profile",
    "export",
    "train",
    "datasets",
    "evaluate",
    "path",
]

command.root()
sys.exit(0)
