r"""
Weight surgeon CLI.
"""

from __future__ import annotations

import argparse
import typing as T

import pandas as pd
import regex as re
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
        "--match", "-m", type=re.compile, default=None, help="filters weights by name"
    )
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
            if args.match and not args.match.search(name):
                continue
            rec = {
                "weight": name,
                "shape": tuple(weight.shape),
                "dtype": str(weight.dtype).split(".")[-1],
                "min": weight.min().item(),
                "max": weight.max().item(),
                "mean": (
                    weight.mean().item() if torch.is_floating_point(weight) else None
                ),
                "std": weight.std().item() if torch.is_floating_point(weight) else None,
            }
            data.append(rec)
        data = pd.DataFrame(data)
        print(up.log.create_table(data, format="wide"))


class SubsetCommand(Subcommand, name="subset"):
    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument("--output", "-o", type=up.file_io.Path, help="output file")
        _add_weights_arg(prs)

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        res = {}
        for name, weight in args.weights.items():
            if args.match and not args.match.search(name):
                continue
            res[name] = weight

        for k in res.keys():
            print(k)

        if not args.output:
            print(
                "\n(provide an output path with `--output` to save the altered weights)"
            )
            return

        if args.output.suffix.lower() == ".safetensors":
            safetensors.torch.save_file(res, args.output)
        elif args.output.suffix.lower() in (".pth", ".pt"):
            torch.save(res, args.output)
        else:
            msg = f"Unsupported file format: {args.output}"
            raise ValueError(msg)

        print(f"\n (saved to {str(args.output)!r})")


command_name = up.file_io.Path(__file__).stem
command(command_name, help="model weight surgeon")(Subcommand)
if __name__ == "__main__":
    command.root(command_name)
