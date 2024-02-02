"""
Entry point for the CLI.
"""

from __future__ import annotations

import sys

from unipercept.cli import backbones, command, echo, profile, trace, train, datasets

__all__ = ["backbones", "echo", "profile", "trace", "train", "datasets"]

command.root()
sys.exit(0)
