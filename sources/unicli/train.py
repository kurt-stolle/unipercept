"""Training and evaluation entry point."""
from __future__ import annotations
import os
import argparse
import typing as T

import torch
import torch.nn as nn

import unipercept as up
from unipercept.utils.config import LazyConfig, _lazy, templates
from unipercept.utils.logutils import create_table, get_logger

from ._cmd import command

_logger = get_logger(__name__)
_config_t: T.TypeAlias = templates.LazyConfigFile[up.data.DataConfig, up.trainer.Trainer, nn.Module]


@command(help="trian a model", description=__doc__)
@command.with_config
def train(subparser: argparse.ArgumentParser):
    subparser.add_argument("--headless", action="store_true", help="disable all interactive prompts")
    subparser.add_argument(
        "--detect-anomalies",
        action="store_true",
        dest="anomalies",
        default=False,
        help="flag to enable anomaly detection in autograd",
    )
    subparser.add_argument(
        "--evaluation", "-E", action="store_true", help="run in evaluation mode (no training, only evaluation)"
    )
    subparser.add_argument(
        "--no-jit", action="store_true", help="disable JIT compilation"
    )
    options_resume = subparser.add_mutually_exclusive_group(required=False)
    options_resume.add_argument(
        "--reset",
        action="store_true",
        help="whether to reset the output directory and caches if they exist",
    )
    options_resume.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume training from the checkpoint directory",
    )
    subparser.add_argument("--dataloader-train", type=str, default="train", help="name of the train dataloader")
    subparser.add_argument("--dataloader-test", type=str, default="test", help="name of the test dataloader")

    return main


def main(args):
    if args.anomalies:
        _logger.info("Enabling anomaly detection in autograd")
        torch.autograd.set_detect_anomaly(True)
    if args.no_jit:
        _logger.info("Disabling JIT compilation")
        os.environ["PYTORCH_JIT"] = "0"

    config: _config_t = args.config
    trainer: up.trainer.Trainer = _lazy.instantiate(config.trainer)
    config_path = trainer.path / "config.yaml"

    if config_path.exists():
        _logger.info(
            "A serialized config YAML file exists at %s. Recovering trainer at latest checkpoint.",
            config_path,
        )
        trainer.recover()
    elif trainer._xlr.is_main_process:
        _logger.info("Storing serialized config to YAML file %s", trainer.path)
        LazyConfig.save(config, str(config_path))

    loaders: dict[str, up.data.DataLoaderFactory] = _lazy.instantiate(args.config.data.loaders)

    if args.evaluation:
        results = trainer.evaluate(
            lambda _: _lazy.instantiate(config.model),
            loaders[args.dataloader_test],
        )
        _logger.info("Evaluation results: \n%s", create_table(results, format="long"))
    else:
        trainer.train(
            lambda _: _lazy.instantiate(config.model),
            loaders[args.dataloader_train],
            checkpoint=None,
            trial=None,
            evaluation_loader_factory=loaders[args.dataloader_test],
        )


if __name__ == "__main__":
    command.root("train")
