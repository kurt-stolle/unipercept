"""Training entry point."""
from __future__ import annotations

import argparse
import os
import sys
import typing as T

import torch
from omegaconf import DictConfig
from tabulate import tabulate

import unipercept as up
from unipercept.cli._command import command
from unipercept.cli._config import ConfigFileContentType as config_t

_logger = up.log.get_logger()


KEY_SESSION_ID = "session_id"


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
        help="run in evaluation mode",
    )

    return _main


def _apply_debug_mode(lazy_config: config_t) -> None:
    os.environ["WANDB_OFFLINE"] = "true"
    torch.autograd.set_detect_anomaly(True)
    lazy_config.ENGINE.params.full_determinism = True


def _step(args) -> config_t:
    if args.no_jit:
        _logger.info("Disabling JIT compilation")
        os.environ["PYTORCH_JIT"] = "0"
    if args.debug:
        _apply_debug_mode(args.config)

    return args.config


def _main(args):
    lazy_config: config_t = _step(args)

    up.state.barrier()  # Ensure the config file is not created before all processes validate its existence

    engine: up.engine.Engine = up.config.instantiate(lazy_config.ENGINE)

    if args.resume:
        session_id_recovered = lazy_config[KEY_SESSION_ID]
        if session_id_recovered is None:
            msg = "Training configuration does not support resumption"
            raise ValueError(msg)

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

        engine.session_id = session_id_recovered
        engine.session_dir = resume_dir
    elif up.state.check_main_process():
        _logger.info("Storing serialized config to YAML file %s", engine.config_path)
        lazy_config[KEY_SESSION_ID] = engine.session_id
        engine.config = lazy_config

    _logger.info(
        "Starting engine session:\n%s",
        tabulate(
            [
                ("Session ID", engine.session_id),
                ("Session path", str(engine.session_dir)),
            ],
        ),
    )

    # Setup dataloaders
    model_factory = up.model.ModelFactory(lazy_config.MODEL, weights=args.weights)
    try:
        if args.evaluation:
            _logger.info("Running in EVALUATION ONLY MODE. Be aware that no weights are loaded if not provided explicitly!")
            results = engine.run_evaluation(model_factory)
            _logger.info(
                "Evaluation results: \n%s", up.log.create_table(results, format="long")
            )
        else:
            results = engine.run_training_procedure(
                model_factory, max(args.stage, 0), weights=args.weights or None
            )
    except KeyboardInterrupt:
        output_path = up.file_io.Path("//output/").resolve()
        config_path = engine.config_path.resolve()
        if config_path.is_relative_to(output_path):
            config_path = config_path.relative_to(output_path).as_posix()
            config_path = f"//output/{config_path}"

        print("\n", flush=True, file=sys.stdout)
        print("\n", flush=True, file=sys.stderr)

        _logger.info(
            "Training interrupted by user. To resume, use the --resume flag and configuration file: %s",
            engine.config_path,
        )


if __name__ == "__main__":
    command.root("train")
