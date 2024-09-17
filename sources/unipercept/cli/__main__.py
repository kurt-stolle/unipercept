"""
Entry point for the CLI.
"""

from __future__ import annotations

import importlib
import sys

from ._command import command


def _import_cmd(cmd: str):
    importlib.import_module(f"unipercept.cli.{cmd}")


def _import_all():
    for cmd in (
        "backbones",
        "check",
        "datasets",
        "du",
        "echo",
        "evaluate",
        "export",
        "path",
        "profile",
        "ros2",
        "run",
        "surgeon",
        "show",
        "train",
    ):
        _import_cmd(cmd)
    command.root()


if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
    _import_cmd(sys.argv[1])
else:
    _import_all()

command.root()
sys.exit(0)
