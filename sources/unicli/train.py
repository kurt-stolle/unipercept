"""Training and evaluation entry point."""
from __future__ import annotations

import argparse
import dataclasses
import typing as T

import torch.nn as nn
from uniutils.config import _lazy, templates
from uniutils.logutils import get_logger

import unipercept as up

from ._cmd import command

_logger = get_logger(__name__)
_config_t: T.TypeAlias = templates.LazyConfigFile[up.data.DataConfig, up.trainer.Trainer, up.modeling.PerceptionModel]


@dataclasses.dataclass
class ModelFactory:
    lazy_model: _lazy.LazyObject[up.modeling.PerceptionModel]

    def __call__(self, trial: T.Any = None) -> up.modeling.PerceptionModel:
        return _lazy.instantiate(self.lazy_model)


def main(args):
    config: _config_t = args.config
    model_factory = ModelFactory(config.model)
    trainer: up.trainer.Trainer = _lazy.instantiate(config.trainer)
    loaders: dict[str, up.data.DataLoaderFactory] = _lazy.instantiate(args.config.data.loaders)

    result = trainer.train(model_factory, loaders["train"], "best", trial=None)

    _logger.info(result)


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
