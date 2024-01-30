"""Training and evaluation entry point."""
from __future__ import annotations

import argparse
import os
import typing as T

import torch
from omegaconf import DictConfig

import unipercept as up
from unipercept.cli._command import command

_logger = up.log.get_logger()
_config_t: T.TypeAlias = DictConfig


@command(help="trian a model", description=__doc__)
@command.with_config
def train(p: argparse.ArgumentParser):
    # General arguments
    p.add_argument(
        "--headless", action="store_true", help="disable all interactive prompts"
    )
    p.add_argument("--no-jit", action="store_true", help="disable JIT compilation")
    p.add_argument(
        "--stage", type=int, default=-1, help="stage number to start training from"
    )
    p.add_argument("--weights", "-w", type=str, help="path to load model weights from")
    p.add_argument(
        "--debug",
        action="store_true",
        help="optimizes the configuration for model debugging",
    )
    p.add_argument(
        "--resume",
        "-R",
        action="store_true",
        help="continue training from the last checkpoint",
    )

    # Mode (training/evaluation/...)
    p_mode = p.add_mutually_exclusive_group(required=False)
    p_mode.add_argument(
        "--training", "-T", action="store_true", help="run in training mode (default)"
    )
    p_mode.add_argument(
        "--evaluation",
        "-E",
        action="store_true",
        help="run in evaluation mode (no training, only evaluation)",
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


def main(args):
    lazy_config: _config_t = setup(args)

    up.state.barrier()  # Ensure the config file is not created before all processes validate its existence

    engine: up.engine.Engine = up.config.instantiate(lazy_config.engine)

    if args.resume:
        if not args.config_path.endswith(".yaml"):
            msg = "Cannot resume training without a YAML config file"
            raise ValueError(msg)
        resume_dir = args.config_path.parent
        if not (resume_dir / "outputs").exists():
            msg = "Cannot resume training from a directory without outputs"
            raise ValueError(msg)
        if not (resume_dir / "logs").exists():
            msg = "Cannot resume training from a directory without logs"
            raise ValueError(msg)

        engine.session_dir = resume_dir

    if up.state.check_main_process():
        _logger.info("Storing serialized config to YAML file %s", engine.config_path)
        up.config.save_config(lazy_config, str(engine.config_path))

    # Setup dataloaders
    model_factory = up.model.ModelFactory(lazy_config.model)

    if args.evaluation:
        results = engine.run_evaluation(model_factory, weights=args.weights)
        _logger.info(
            "Evaluation results: \n%s", up.log.create_table(results, format="long")
        )
    else:
        engine.run_training(
            model_factory,
            trial=None,
            stage=args.stage if args.stage >= 0 else None,
            weights=args.weights,
        )


if __name__ == "__main__":
    command.root("train")
