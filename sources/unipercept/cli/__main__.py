"""
Entry point for the CLI.
"""

from __future__ import annotations

import sys

from unipercept.cli import backbones, command, echo, profile, trace, train

__all__ = ["backbones", "echo", "profile", "trace", "train"]

command.root()
sys.exit(0)
