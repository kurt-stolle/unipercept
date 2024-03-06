"""Evaluation entry point."""

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


@command(help="evaluate a trained model", description=__doc__)
@command.with_config
def evaluate(p: argparse.ArgumentParser):
    p.add_argument("--no-jit", action="store_true", help="disable JIT compilation")
    p.add_argument(
        "--debug",
        action="store_true",
        help="optimizes the configuration for model debugging",
    )
    p.add_argument(
        "--weights",
        "-w",
        type=str,
        help="path to load model weights from, if not are inferred from the configuration path",
    )
    p.add_argument(
        "--path",
        "-o",
        type=up.file_io.Path,
        help="path to store outputs from evaluation",
    )
    p.add_argument(
        "--suite",
        nargs="+",
        type=str,
        help="evaluation suite to run",
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

    up.state.barrier()  # Ensure the config file is not created before all processes validate its existence
    return args.config


def _main(args):
    config = _step(args)
    engine = up.create_engine(config)
    model_factory = up.create_model_factory(config, weights=args.weights or None)
    suites = args.suite if args.suite is not None and len(args.suite) > 0 else None
    try:
        results = engine.run_evaluation(
            model_factory,
            suites=suites,
            path=up.file_io.Path(args.path) if args.path is not None else None,
        )
        _logger.info(
            "Evaluation results: \n%s", up.log.create_table(results, format="long")
        )
    except KeyboardInterrupt:
        _logger.info("Evaluation interrupted")


if __name__ == "__main__":
    command.root("train")
