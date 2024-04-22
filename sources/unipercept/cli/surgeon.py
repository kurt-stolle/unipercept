r"""
Weight surgeon CLI.
"""

from __future__ import annotations
import argparse
import safetensors
import torch
import typing_extensions as TX
import typing as T
from unipercept.cli._command import command, logger

import unipercept as up

__all__ = []

Subcommand = up.utils.cli.create_subtemplate()


class StatsSubcommand(Subcommand, name="stats"):
    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument(
            "weights",
            type=up.file_io.Path,
            help="model state dict file, e.g. `model.pth` or `model.safetensors`",
        )

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        match args.weights.suffix.lower():
            case ".safetensors":
                state = safetensors.load(args.weights)
            case ".pth":
                state = torch.load(args.weights, map_location="cpu")
            case _:
                msg = f"Unsupported file format: {args.weights}"
                raise ValueError(msg)

def _read_state_dict(weights: up.file_io.Path) -> T.Any:
    match weights.suffix.lower():
        case ".safetensors":
            return safetensors.load(weights)
        case ".pth":
            return torch.load(weights, map_location="cpu")
        case _:
            msg = f"Unsupported file format: {weights}"
            raise ValueError(msg)

command_name = up.file_io.Path(__file__).stem
command(command_name, help="model weight surgeon")(Subcommand)
if __name__ == "__main__":
    command.root(command_name)
