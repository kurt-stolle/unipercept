"""
Entry point for the CLI.
"""

from __future__ import annotations

import sys

from unipercept.cli import backbones, command, datasets, echo, profile, trace, train, evaluate

__all__ = ["backbones", "echo", "profile", "trace", "train", "datasets", "evaluate"]

command.root()
sys.exit(0)
