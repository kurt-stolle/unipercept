"""
Trace command. This command is used to profile a model using FX tracing.
"""

from __future__ import annotations

import argparse

import torch.fx

import unipercept as up
from unipercept.cli._command import command, logger


@command(help="analyse a model using FX tracing", description=__doc__)
@command.with_config
def export(subparser: argparse.ArgumentParser):
    mode = subparser.add_mutually_exclusive_group()
    mode.add_argument("--training", "-T", action="store_true", help="export training")
    mode.add_argument(
        "--inference", "-I", action="store_true", help="export inference", default=True
    )
    subparser.add_argument(
        "--symbolic", action="store_true", help="use symbolic tracing", default=False
    )
    subparser.add_argument(
        "path", type=str, nargs="*", help="path to submodule to profile"
    )

    return _main


def _export_symbolic(args, model):
    gm = torch.fx.symbolic_trace(model)
    gm.print_readable()
    gm.graph.print_tabular()


def _export_trace(args, model):
    dataloader, info = up.create_dataset(args.config, return_loader=False)
    inputs = next(dataloader)

    exp = torch.export.export(model, (inputs,), strict=False)

    print(exp)


def _main(args):
    model = up.create_model(args.config)

    if args.training:
        logger.info("Exporting training graph")
        model = model.train()
    elif args.inference:
        logger.info("Exporting inference graph")
        model = model.eval()
    else:
        msg = "Expected either training or inference mode!"
        raise NotImplementedError(msg)

    submodule = model
    for name in args.path:
        submodule = getattr(submodule, name)

    if not isinstance(submodule, torch.nn.Module):
        full_path = ".".join(args.path)
        msg = (
            f"Submodule model.{full_path} is not a torch.nn.Module! "
            f"Got: {type(submodule)}"
        )
        raise ValueError(msg)

    logger.info("Running symbolic tracing on ", submodule)

    if args.symbolic:
        _export_symbolic(args, submodule)
    else:
        _export_trace(args, submodule)


if __name__ == "__main__":
    command.root("trace")
