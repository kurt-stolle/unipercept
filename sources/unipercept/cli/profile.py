"""
Model profiling entry point.
"""
from __future__ import annotations

import argparse
import typing as T

import torch
import torch.autograd
import torch.autograd.profiler
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from unipercept import create_dataset, create_engine, create_model, file_io
from unipercept.cli._command import command
from unipercept.log import get_logger
from unipercept.model import ModelAdapter
from unipercept.utils.time import get_timestamp


@command(help="trian a model", description=__doc__)
@command.with_config
def profile(subparser: argparse.ArgumentParser):
    subparser.add_argument(
        "--loader", "-l", type=str, default="train", help="loader to use for profiling"
    )
    subparser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=3,
        help="number of iterations to profile",
    )
    subparser.add_argument(
        "--sort-by",
        type=str,
        default="self_cpu_memory_usage",
        help="sort by this column when showing output in console",
    )
    subparser.add_argument(
        "--memory",
        action="store_true",
        help="profile using CUDA memory manager, emitting a snapshot.",
        default=False,
    )
    subparser.add_argument(
        "--flops",
        action="store_true",
        help="profile FLOPs using methodology proposed by FAIR's `fvcore` package.",
        default=False,
    )
    subparser.add_argument(
        "--trace",
        action="store_true",
        help="profile using torch profiler, emitting a Chrome trace",
        default=False,
    )
    subparser.add_argument("--weights", "-w", default=None, type=str)

    mode = subparser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--training", "-T", action="store_true", help="profile training")
    mode.add_argument(
        "--inference", "-I", default=True, action="store_true", help="profile inference"
    )

    subparser.add_argument("path", nargs="*", type=str)

    return _main


_logger = get_logger(__name__)


def _find_session_path(config: T.Any) -> file_io.Path:
    """
    Find the path to the session file.
    """
    try:
        path = file_io.Path(
            f"//output/{config.engine.params.project_name}/{config.engine.params.session_name}/profile"
        )
    except KeyError:
        path = file_io.Path(f"//output/uncategorized/{get_timestamp()}/profile")
        _logger.warning("No session file found in config, using default path")

    path.mkdir(exist_ok=True, parents=True)

    return path


def _analyse_flops(model: nn.Module, loader: torch.utils.data.DataLoader) -> None:
    from fvcore.nn import FlopCountAnalysis

    inputs = next(iter(loader))
    model_adapter = ModelAdapter(model, inputs, allow_non_tensor=True)

    flops = FlopCountAnalysis(model_adapter, inputs=model_adapter.flattened_inputs)

    _logger.info("Running FLOP analysis...")
    _logger.info("Total FLOPs:\n%s", flops.total())


def _analyse_memory(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    handler,
    *,
    iterations: int,
    path_export: file_io.Path,
) -> None:
    """ """

    _logger.info("Profiling model with snapshot")
    torch.cuda.memory._record_memory_history()

    loader_iter = iter(loader)
    for _ in tqdm(range(iterations)):
        data = next(loader_iter)
        handler(model, data)

    path_snapshot = path_export / "cuda_snapshot.pkl"
    torch.cuda.memory._dump_snapshot(str(path_snapshot))

    _logger.info(
        "Upload the snapshot to https://pytorch.org/memory_viz using local path: %s",
        path_snapshot,
    )


def _analyse_trace(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    handler,
    *,
    iterations: int,
    path_export: file_io.Path,
) -> None:
    _logger.info("Profiling model")

    loader_iter = iter(loader)

    with torch.profiler.profile(
        profile_memory=True,
        with_flops=True,
        with_stack=True,
        with_modules=True,
        record_shapes=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(path_export)),
    ) as prof:
        for _ in tqdm(iterable=range(iterations)):
            data = next(loader_iter)
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


def _main(args):
    config: T.Any = args.config
    path_export = _find_session_path(config)

    _logger.info("Saving results to %s", path_export)

    engine = create_engine(config)
    model = create_model(config, state=args.weights)
    model.to(engine.device)

    if args.training:
        handler = engine.run_training_step
        model.train()
        torch.inference_mode(True)
    elif args.inference:
        handler = engine.run_inference_step
        model.eval()
        torch.inference_mode(False)
    else:
        _logger.error(
            "Unknown mode; provide either the `--training` or `--inference` flag"
        )
        exit(1)

    _logger.info("Preparing dataset")
    loader, info = create_dataset(config, variant=args.loader, batch_size=1)

    if not any([args.flops, args.memory, args.trace]):
        _logger.info("No profiling method specified; exiting")
        exit(0)
    if args.flops:
        _analyse_flops(model, loader)
    if args.memory:
        _analyse_memory(
            model, loader, handler, iterations=args.iterations, path_export=path_export
        )
    if args.trace:
        _analyse_trace(
            model, loader, handler, iterations=args.iterations, path_export=path_export
        )


if __name__ == "__main__":
    command.root("profile")
