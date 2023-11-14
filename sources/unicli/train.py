"""Training and evaluation entry point."""
from __future__ import annotations

import argparse
import typing as T

import torch.nn as nn

import unipercept as up
from unipercept.utils.config import LazyConfig, _lazy, templates
from unipercept.utils.logutils import get_logger

from ._cmd import command

_logger = get_logger(__name__)
_config_t: T.TypeAlias = templates.LazyConfigFile[up.data.DataConfig, up.trainer.Trainer, nn.Module]


def main(args):
    cfg: _config_t = args.config

    # Print a Python representation of the config
    # _logger.info(f"Configuration:\n{LazyConfig.to_py(cfg)}")

    # Materialize trainer
    trainer: up.trainer.Trainer = _lazy.instantiate(cfg.trainer)

    # Save configuration for layer use and inspection
    if trainer._xlr.is_main_process:
        config_serialized_path = trainer.path / "config.yaml"

        if config_serialized_path.exists():
            _logger.info(
                "A serialized config YAML file exists at %s. Recovering trainer at latest checkpoint.",
                config_serialized_path,
            )
            trainer.recover()
        else:
            _logger.info("Storing serialized config to YAML file %s", trainer.path)
            LazyConfig.save(cfg, str(config_serialized_path))

    # Materialize loaders
    loaders: dict[str, up.data.DataLoaderFactory] = _lazy.instantiate(args.config.data.loaders)

    # Start training
    trainer.train(
        lambda _: _lazy.instantiate(cfg.model),
        loaders["train"],
        checkpoint=None,
        trial=None,
        evaluation_loader_factory=loaders["test"],
    )


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
    return main


if __name__ == "__main__":
    command.root("train")
