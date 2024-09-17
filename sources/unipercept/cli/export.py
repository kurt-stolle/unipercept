"""
Trace command. This command is used to profile a model using FX tracing.
"""

from __future__ import annotations

import argparse
import os

import torch.fx

import unipercept as up
from unipercept.cli._command import command, logger

os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"


@command(help="analyse a model using FX tracing", description=__doc__)
@command.with_config
def export(subparser: argparse.ArgumentParser):
    mode = subparser.add_mutually_exclusive_group()
    mode.add_argument("--training", "-T", action="store_true", help="export training")
    mode.add_argument(
        "--inference", "-I", action="store_true", help="export inference", default=True
    )

    exporter = subparser.add_mutually_exclusive_group()
    exporter.add_argument("--fx", action="store_true", help="use FX symbolic tracing")
    exporter.add_argument(
        "--onnx", action="store_true", help="use ONNX with JIT tracing"
    )

    subparser.add_argument(
        "path", type=str, nargs="*", help="path to submodule to profile"
    )

    return _main


def _export_symbolic(args, model):
    gm = torch.fx.symbolic_trace(model)
    gm.print_readable()
    gm.graph.print_tabular()


def _export_onnx(args, model):
    dataloader, info = up.create_dataset(args.config, return_loader=False)
    inputs = next(dataloader)

    inputs = tuple(model.select_inputs(inputs, torch.device("cuda")))
    model = model.cuda()

    adapter = up.model.ModelAdapter(model, inputs)
    exp = torch.onnx.export(
        adapter, adapter.flattened_inputs, "model.onnx", verbose=True
    )

    print(exp)


def _export_native(args, model):
    dataloader, info = up.create_dataset(args.config, return_loader=False)
    inputs = next(dataloader)

    inputs = tuple(model.select_inputs(inputs, torch.device("cuda")))
    model = model.cuda()

    try:
        exp = torch.export.export(model, inputs)
    except Exception as e:
        print(e)
        exp = torch.export.export(model, inputs, strict=False)

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

    if args.fx:
        _export_symbolic(args, submodule)
    elif args.onnx:
        _export_onnx(args, submodule)
    else:
        _export_native(args, submodule)


if __name__ == "__main__":
    command.root("trace")
