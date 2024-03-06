"""
Entry point for the CLI.
"""

from __future__ import annotations

import sys

from unipercept.cli import (
    backbones,
    command,
    datasets,
    echo,
    evaluate,
    export,
    profile,
    train,
)

__all__ = ["backbones", "echo", "profile", "export", "train", "datasets", "evaluate"]

command.root()
sys.exit(0)
