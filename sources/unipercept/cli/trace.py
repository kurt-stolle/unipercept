"""
Trace command. This command is used to profile a model using FX tracing.
"""

from __future__ import annotations

import argparse

import torch.fx

from unipercept import create_model
from unipercept.cli._command import command


@command(help="analyse a model using FX tracing", description=__doc__)
@command.with_config
def trace(subparser: argparse.ArgumentParser):
    mode = subparser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--training", "-T", action="store_true", help="profile training")
    mode.add_argument(
        "--inference", "-I", action="store_true", help="profile inference"
    )

    subparser.add_argument(
        "path", type=str, nargs="*", help="path to submodule to profile"
    )

    return _main


def _main(args):
    model = create_model(args.config)

    if args.training:
        model = model.train()
    elif args.inference:
        model = model.eval()
    else:
        raise ValueError("Expected either training or inference mode!")

    submodule = model
    for name in args.path:
        submodule = getattr(submodule, name)

    if not isinstance(submodule, torch.nn.Module):
        full_path = ".".join(args.path)
        raise ValueError(
            f"Submodule model.{full_path} is not a torch.nn.Module! Got: {type(submodule)}"
        )

    print("Running symbolic tracing on ", submodule)

    gm = torch.fx.symbolic_trace(submodule)

    gm.print_readable()

    gm.graph.print_tabular()


if __name__ == "__main__":
    command.root("trace")
