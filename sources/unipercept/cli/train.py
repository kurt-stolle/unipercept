"""Training and evaluation entry point."""
from __future__ import annotations

import argparse
import os
import typing as T

import accelerate
import torch
import torch.nn as nn
from unicore import file_io

import unipercept as up

from ._command import command

_logger = up.log.get_logger(__name__)
_config_t: T.TypeAlias = up.config.templates.LazyConfigFile[up.data.DataConfig, up.engine.Engine, nn.Module]


@command(help="trian a model", description=__doc__)
@command.with_config
def train(subparser: argparse.ArgumentParser):
    subparser.add_argument("--headless", action="store_true", help="disable all interactive prompts")
    subparser.add_argument(
        "--evaluation", "-E", action="store_true", help="run in evaluation mode (no training, only evaluation)"
    )
    subparser.add_argument("--no-jit", action="store_true", help="disable JIT compilation")
    # options_resume = subparser.add_mutually_exclusive_group(required=False)
    # options_resume.add_argument(
    #     "--reset",
    #     action="store_true",
    #     help="whether to reset the output directory and caches if they exist",
    # )
    # options_resume.add_argument(
    #     "--resume",
    #     action="store_true",
    #     help="whether to attempt to resume training from the checkpoint directory",
    # )
    subparser.add_argument("--stage", type=int, default=-1, help="stage number to start training from")
    subparser.add_argument("--weights", "-w", type=str, help="path to load model weights from")

    subparser_mode = subparser.add_mutually_exclusive_group(required=False)
    subparser_mode.add_argument(
        "--development",
        action="store_true",
        help="optimizes the configuration for library development, implies --debug",
    )
    subparser_mode.add_argument(
        "--debug",
        action="store_true",
        help="optimizes the configuration for model debugging",
    )

    return main


def apply_debug_mode(lazy_config: _config_t) -> None:
    torch.autograd.set_detect_anomaly(True)

    lazy_config.engine.params.full_determinism = True


def apply_development_mode(lazy_config: _config_t) -> None:
    os.environ["WANDB_OFFLINE"] = "true"

    apply_debug_mode(lazy_config)


def setup(args) -> _config_t:
    # Disable JIT
    if args.no_jit:
        _logger.info("Disabling JIT compilation")
        os.environ["PYTORCH_JIT"] = "0"

    # Dev/Debug mode
    if args.development:
        apply_development_mode(args.config)
    elif args.debug:
        apply_debug_mode(args.config)

    return args.config


def main(args):
    lazy_config: _config_t = setup(args)

    # Before creating the engine across processes, check if the config file exists and recover the engine if it does.
    lazy_engine = lazy_config.engine
    lazy_engine.params = up.config.instantiate(lazy_engine.params)
    config_path = file_io.Path(lazy_engine.params.root) / "config.yaml"
    do_recover = config_path.exists()

    state = accelerate.PartialState()
    state.wait_for_everyone()  # Ensure the config file is not created before all processes validate its existence

    engine: up.engine.Engine = up.config.instantiate(lazy_config.engine)

    if do_recover:
        _logger.info(
            "A serialized config YAML file exists at %s. Recovering engine at latest checkpoint.",
            config_path,
        )
        engine.recover()
    elif state.is_main_process:
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
