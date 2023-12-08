"""
Entry point for the CLI.
"""

from __future__ import annotations

import sys

from unipercept.cli import command

command.root()
sys.exit(0)
