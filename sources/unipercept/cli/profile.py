"""
Model profiling entry point.
"""
from __future__ import annotations

import argparse
import time
import typing as T
from pprint import pformat

import torch
import torch.autograd
import torch.autograd.profiler
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from unicore import file_io

import unipercept as up
from unipercept.config import templates
from unipercept.log import get_logger
from unipercept.utils.time import get_timestamp

from ._command import command

_logger = get_logger(__name__)
_config_t: T.TypeAlias = templates.LazyConfigFile[up.data.DataConfig, up.engine.Engine, nn.Module]


def _step_training(model: nn.Module, data: up.model.InputData) -> None:
    """
    Run a training step on the model, i.e. forward and backward pass.
    """
    outs = model(data)
    loss = torch.stack(list(outs.losses.values()))
    grad = torch.ones_like(loss)

    loss.backward(gradient=grad)


def _step_inference(model: nn.Module, data: up.model.InputData) -> None:
    """
    Run an inference step on the model, i.e. forward pass only.
    """
    model(data)


def _fetch_data(loader: T.Iterator[up.model.InputData], device: str, fp16: bool) -> up.model.InputData:
    """
    Fetch the next batch from the loader.
    """
    data = next(loader).to(device=device)  # type: ignore
    if fp16:
        data.captures.images = data.captures.images.half()
    return data


def _find_session_path(config: _config_t) -> file_io.Path:
    """
    Find the path to the session file.
    """
    try:
        path = file_io.Path(f"//output/{config.engine.params.project_name}/{config.engine.params.session_name}/profile")
    except KeyError:
        path = file_io.Path(f"//output/uncategorized/{get_timestamp()}/profile")
        _logger.warning("No session file found in config, using default path")

    path.mkdir(exist_ok=True, parents=True)

    return path


def _prepare_model(config: _config_t, device: str, fp16: bool, training: bool) -> nn.Module:
    """
    Prepare the model for profiling.
    """
    _logger.info("Creating model")
    model = up.create_model(config, device=device)
    if training:
        _logger.info("Setting model to training mode")
        model.train()
    else:
        _logger.info("Setting model to inference mode")
        model.eval()
    if fp16:
        model.half()
    return model


def main(args):
    # Read parsed configuration
    config: _config_t = args.config

    # Print the configuration
    _logger.info("Configuration:\n%s", pformat(up.config.flatten_config(config)))

    # Save the results
    path_export = _find_session_path(config)

    _logger.info("Saving results to %s", path_export)

    # Create the model
    if args.training:
        training = True
        handler = _step_training
    elif args.inference:
        training = False
        handler = _step_inference
    else:
        _logger.error("Unknown mode; provide either the `--training` or `--inference` flag")
        exit(1)
    model = _prepare_model(config, args.device, args.fp16, training=training)

    # Get the loader
    _logger.info("Preparing dataset")
    loader, info = up.create_dataset(config, variant=args.loader, batch_size=1)

    # Profile with snapshot
    _logger.info("Profiling model with snapshot")
    torch.cuda.memory._record_memory_history()

    for _ in tqdm(range(args.iterations)):
        data = _fetch_data(loader, args.device, args.fp16)
        handler(model, data)

    torch.cuda.memory._dump_snapshot(str(path_export / "cuda_snapshot.pkl"))

    # Run the profiler
    _logger.info("Profiling model")
    with torch.profiler.profile(
        profile_memory=True,
        with_flops=True,
        with_stack=True,
        with_modules=True,
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(path_export)),
    ) as prof:
        for _ in tqdm(range(args.iterations)):
            data = _fetch_data(loader, args.device, args.fp16)
            handler(model, data)
            prof.step()
    assert prof is not None

    # Print the results
    _logger.info(
        "Key averages sorted by `self_cuda_time_total`:\n\n%s\n",
        prof.key_averages().table(sort_by=args.sort_by, row_limit=-1),
    )

    with open(path_export / "key_averages.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_stacks(str(path_export / "stacks.txt"))

    _logger.info("Exporting chrome trace file...")
    prof.export_chrome_trace(str(path_export / "trace.json"))

    _logger.debug("Finished profiling session.")


@command(help="trian a model", description=__doc__)
@command.with_config
def profile(subparser: argparse.ArgumentParser):
    subparser.add_argument("--device", "-d", type=str, default="cuda", help="device to use for training")
    subparser.add_argument("--loader", "-l", type=str, default="train", help="loader to use for profiling")
    subparser.add_argument("--iterations", "-i", type=int, default=3, help="number of iterations to profile")
    subparser.add_argument("--fp16", action="store_true", help="use fp16")
    subparser.add_argument(
        "--sort-by",
        type=str,
        default="self_cpu_memory_usage",
        help="sort by this column when showing output in console",
    )
    subparser.add_argument("--memory", "-m", action="store_true", help="profile memory", default=True)

    mode = subparser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--training", "-T", action="store_true", help="profile training")
    mode.add_argument("--inference", "-I", action="store_true", help="profile inference")

    return main


if __name__ == "__main__":
    command.root("profile")
