"""Training and evaluation entry point."""
from __future__ import annotations

import argparse
import os
import typing as T

import torch
import torch.nn as nn
from unicore import file_io

import unipercept as up

from ._command import command

_logger = up.log.get_logger(__name__)
_config_t: T.TypeAlias = up.config.templates.LazyConfigFile[up.data.DataConfig, up.engine.Engine, nn.Module]


@command(help="trian a model", description=__doc__)
@command.with_config
def train(p: argparse.ArgumentParser):
    # General arguments
    p.add_argument("--headless", action="store_true", help="disable all interactive prompts")
    p.add_argument("--no-jit", action="store_true", help="disable JIT compilation")
    p.add_argument("--stage", type=int, default=-1, help="stage number to start training from")
    p.add_argument("--weights", "-w", type=str, help="path to load model weights from")
    p.add_argument(
        "--debug",
        action="store_true",
        help="optimizes the configuration for model debugging",
    )

    # Mode (training/evaluation/...)
    p_mode = p.add_mutually_exclusive_group(required=False)
    p_mode.add_argument("--training", "-T", action="store_true", help="run in training mode (default)")
    p_mode.add_argument(
        "--evaluation", "-E", action="store_true", help="run in evaluation mode (no training, only evaluation)"
    )

    return main


def apply_debug_mode(lazy_config: _config_t) -> None:
    os.environ["WANDB_OFFLINE"] = "true"
    torch.autograd.set_detect_anomaly(True)
    lazy_config.engine.params.full_determinism = True


def setup(args) -> _config_t:
    if args.no_jit:
        _logger.info("Disabling JIT compilation")
        os.environ["PYTORCH_JIT"] = "0"
    if args.debug:
        apply_debug_mode(args.config)

    return args.config


# def launch_simple(args):
#     cmd, current_env = prepare_simple_launcher_cmd_env(args)

#     process = subprocess.Popen(cmd, env=current_env)
#     process.wait()
#     if process.returncode != 0:
#         if not args.quiet:
#             raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
#         else:
#             sys.exit(1)


# def launch_cuda(args):
#     from unipercept.utils.cuda import has_p2pib_support
#     import torch.distributed.run as distrib_run

#     current_env = prepare_multi_gpu_env(args)
#     if not has_p2pib_support():
#         current_env["NCCL_P2P_DISABLE"] = "1"
#         current_env["NCCL_IB_DISABLE"] = "1"

#     debug = getattr(args, "debug", False)
#     args = _filter_args(
#         args,
#         distrib_run.get_args_parser(),
#         ["--training_script", args.training_script, "--training_script_args", args.training_script_args],
#     )
#     with patch_environment(**current_env):
#         try:
#             distrib_run.run(args)
#         except Exception:
#             if is_rich_available() and debug:
#                 console = get_console()
#                 console.print("\n[bold red]Using --debug, `torch.distributed` Stack Trace:[/bold red]")
#                 console.print_exception(suppress=[__file__], show_locals=False)
#             else:
#                 raise


def main(args):
    lazy_config: _config_t = setup(args)

    # Before creating the engine across processes, check if the config file exists and recover the engine if it does.
    lazy_engine = lazy_config.engine
    lazy_engine.params = up.config.instantiate(lazy_engine.params)
    config_path = file_io.Path(lazy_engine.params.root) / "config.yaml"
    do_recover = config_path.exists()

    up.state.barrier()  # Ensure the config file is not created before all processes validate its existence

    engine: up.engine.Engine = up.config.instantiate(lazy_config.engine)

    if do_recover:
        _logger.info(
            "A serialized config YAML file exists at %s. Recovering engine at latest checkpoint.",
            config_path,
        )
        engine.recover()
    elif up.state.check_main_process:
        _logger.info("Storing serialized config to YAML file %s", engine.root_path)
        up.config.LazyConfig.save(lazy_config, str(config_path))

    # Setup dataloaders
    model_factory = up.model.ModelFactory(lazy_config.model)

    if args.evaluation:
        results = engine.evaluate(model_factory, weights=args.weights)
        _logger.info("Evaluation results: \n%s", up.log.create_table(results, format="long"))
    else:
        engine.train(model_factory, trial=None, stage=args.stage if args.stage >= 0 else None, weights=args.weights)


if __name__ == "__main__":
    command.root("train")
