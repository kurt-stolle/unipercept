"""
Model profiling entry point.
"""

from __future__ import annotations

import argparse
import inspect
import typing as T

import torch
import torch.autograd
import torch.autograd.profiler
import torch.utils.data
from omegaconf.errors import ConfigAttributeError
from tensordict import TensorDictBase
from torch import nn
from tqdm import tqdm

from unipercept import create_dataset, create_engine, create_model, file_io
from unipercept.cli._command import command
from unipercept.log import logger
from unipercept.model import ModelBase
from unipercept.utils.time import get_timestamp


@command(help="trian a model", description=__doc__)
@command.with_config
def profile(subparser: argparse.ArgumentParser):
    subparser.add_argument(
        "--loader",
        "-l",
        type=str,
        default=None,
        help="evaluation suite key or stage number to use for profiling",
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
        "-f",
        action="store_true",
        help="profile FLOPs using methodology proposed by FAIR's `fvcore` package.",
        default=False,
    )
    subparser.add_argument(
        "--parameter-count",
        "-p",
        action="store_true",
        help="profile model parameters using `fvcore` package.",
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


def _find_session_path(config: T.Any) -> file_io.Path:
    """
    Find the path to the session file.
    """

    proj_name = config.ENGINE.params.project_name

    try:
        path = file_io.Path(f"//output/{proj_name}/{config.session_id}/profile")
    except (KeyError, ConfigAttributeError):
        path = file_io.Path(f"//output/profile/{proj_name}/{get_timestamp()}/profile")
        logger.warning("No session file found in config, using default path: %s", path)

    path.mkdir(exist_ok=True, parents=True)

    return path


def _analyse_params(model: nn.Module, **kwargs) -> None:
    from fvcore.nn import parameter_count_table

    logger.info("Analysing model parameters...")
    logger.info("Parameter count:\n%s", parameter_count_table(model, **kwargs))


def _analyse_flops(
    model: ModelBase,
    loader: torch.utils.data.DataLoader,
    device: torch.types.Device,
    backend="pytorch",
    verbose=False,
    print_per_layer_stat=True,
) -> None:
    from ptflops import get_model_complexity_info

    # from fvcore.nn import FlopCountAnalysis
    # inputs = next(iter(loader))
    # inputs = inputs.to(device)
    # model_adapter = ModelAdapter(model, inputs, allow_non_tensor=True)
    # flops = FlopCountAnalysis(model_adapter, inputs=model_adapter.flattened_inputs)
    # logger.info("Running FLOP analysis...")
    # logger.info("Total FLOPs:\n%s", flops.total())

    model = model.to(device)

    # Get a single batch of data to use as a template
    inputs = next(iter(loader))[:1].to(device)
    inputs_shape = tuple(inputs.captures.images.shape)
    inputs = model.select_inputs(inputs, device)

    # Determine the forward arguments such that we can provide inputs as keywords
    forward_args = inspect.signature(model.forward).parameters

    def randomize(value):
        try:
            return torch.rand_like(value)
        except RuntimeError:
            return torch.rand_like(value.float()).to(value.dtype)

    def inputs_constructor(size: tuple[int, int]) -> dict[str, T.Any]:
        res = {}
        for param, value in zip(forward_args, inputs, strict=False):
            if isinstance(value, torch.Tensor):
                res[param] = randomize(value)
            elif isinstance(value, TensorDictBase):
                res[param] = value.apply(randomize)
            elif value is None:
                res[param] = None
            elif hasattr(value, "clone"):
                res[param] = value.clone()
            else:
                res[param] = value

        return res

    verbose = False

    with device:
        macs, params = get_model_complexity_info(
            model,
            inputs_shape,
            as_strings=True,
            input_constructor=inputs_constructor,
            backend=backend,
            print_per_layer_stat=print_per_layer_stat,
            verbose=verbose,
        )
        if macs is not None:
            print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        if params is not None:
            print("{:<30}  {:<8}".format("Number of parameters: ", params))


def _analyse_memory(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    handler,
    *,
    iterations: int,
    path_export: file_io.Path,
) -> None:
    model = model.cuda()

    logger.info("Warming up...")
    loader_iter = iter(loader)
    for _ in range(11):
        data = next(loader_iter).cuda()
        handler(model, data)

    logger.info("Recording %d iterations...", iterations)
    torch.cuda.memory._record_memory_history()
    for _ in range(iterations):
        data = next(loader_iter).cuda()
        handler(model, data)

    path_snapshot = path_export / "cuda_snapshot.pkl"
    torch.cuda.memory._dump_snapshot(str(path_snapshot))

    logger.info(
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
    logger.info("Profiling model")

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
    logger.info(
        "Key averages sorted by `self_cuda_time_total`:\n\n%s\n",
        prof.key_averages().table(sort_by=args.sort_by, row_limit=-1),
    )

    with open(path_export / "key_averages.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_stacks(str(path_export / "stacks.txt"))

    logger.info("Exporting chrome trace file...")
    prof.export_chrome_trace(str(path_export / "trace.json"))

    logger.debug("Finished profiling session.")


def _main(args):
    config: T.Any = args.config
    path_export = _find_session_path(config)

    logger.info("Saving results to %s", path_export)

    engine = create_engine(config)
    model = create_model(config, state=args.weights)
    model.to(engine.xlr.device)

    if args.parameter_count:
        _analyse_params(model)

    # Exit early when no further profiling is requested
    if not any([args.flops, args.memory, args.trace]):
        exit(0)

    # Prepare model and loader
    if args.training:
        logger.info("Profiling in TRAINING mode")
        handler = engine.run_training_step
        model.train()

        i_stage = int(args.loader) if args.loader is not None else 0
        logger.info("Preparing dataset for stage %d (--loader)", i_stage)

        loader, info = create_dataset(config, variant=i_stage, training=True)
    elif args.inference:
        logger.info("Profiling in EVALUATION mode")
        handler = engine.run_inference_step
        model.eval()

        logger.info("Preparing dataset for evaluation suite %s (--loader)", args.loader)
        loader, info = create_dataset(config, variant=args.loader, batch_size=1)
    else:
        logger.error(
            "Unknown mode; provide either the `--training` or `--inference` flag"
        )
        exit(1)

    if args.flops:
        _analyse_flops(model, loader, device=engine.xlr.device)
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
