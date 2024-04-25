r"""
Weight surgeon CLI.
"""

from __future__ import annotations

import argparse
import typing as T

import pandas as pd
import safetensors.torch
import torch
import typing_extensions as TX

import unipercept as up
from unipercept.cli._command import command, logger

__all__ = []

Subcommand = up.utils.cli.create_subtemplate()


def _add_weights_arg(prs: argparse.ArgumentParser):
    r"""Adds the `weights` argument to the parser."""

    def read_weights(arg: str) -> dict[str, torch.Tensor]:
        path = up.file_io.Path(arg)
        match path.suffix.lower():
            case ".safetensors":
                return safetensors.torch.load_file(path, device="cpu")
            case ".pth":
                return torch.load(path, map_location="cpu")
            case _:
                msg = f"Unsupported file format: {path}"
                raise ValueError(msg)

    prs.add_argument(
        "weights",
        type=read_weights,
        help="model state dict file, e.g. `model.pth` or `model.safetensors`",
    )


class StatsSubcommand(Subcommand, name="stats"):
    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        _add_weights_arg(prs)

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        data = []
        for name, weight in args.weights.items():
            data.append(
                {
                    "weight": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "min": weight.min().item(),
                    "max": weight.max().item(),
                    "mean": weight.mean().item(),
                    "std": weight.std().item(),
                }
            )
        data = pd.DataFrame(data)
        print(up.log.create_table(data))


command_name = up.file_io.Path(__file__).stem
command(command_name, help="model weight surgeon")(Subcommand)
if __name__ == "__main__":
    command.root(command_name)
