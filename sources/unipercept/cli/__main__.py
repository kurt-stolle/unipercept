"""
Entry point for the CLI.
"""

from __future__ import annotations

import sys
import typing as T

import typing_extensions as TX

from unipercept.cli import command

command.root()
sys.exit(0)
