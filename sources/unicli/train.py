"""Training and evaluation entry point."""
from __future__ import annotations
import os
import argparse
import typing as T
import accelerate
import torch
import torch.nn as nn

from unicore import file_io
import os
import unipercept as up
import safetensors
from unipercept.config import LazyConfig, _lazy, templates
from unipercept.log import create_table, get_logger

from ._cmd import command

_logger = get_logger(__name__)
_config_t: T.TypeAlias = templates.LazyConfigFile[up.data.DataConfig, up.engine.Engine, nn.Module]


class ModelFactory:
    def __init__(self, model_config, checkpoint_path: file_io.Path | os.PathLike | str | None = None):
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path 

    def __call__(self, trial: up.engine.Trial | None) -> nn.Module:
        model = T.cast(nn.Module, _lazy.instantiate(self.model_config))

        if self.checkpoint_path is not None:
            _logger.info("Loading model weights from %s", self.checkpoint_path)
            up.load_checkpoint(self.checkpoint_path, model)
        else:
            _logger.info("No model weights checkpoint path provided, skipping recovery")

        return model


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
    subparser.add_argument("--dataloader-train", type=str, default="train", help="name of the train dataloader")
    subparser.add_argument("--dataloader-test", type=str, default="test", help="name of the test dataloader")
    subparser.add_argument("--weights", "-w", type=file_io.Path, help="path to load model weights from")

    return main


def main(args):
    if args.anomalies:
        _logger.info("Enabling anomaly detection in autograd")
        torch.autograd.set_detect_anomaly(True)
    if args.no_jit:
        _logger.info("Disabling JIT compilation")
        os.environ["PYTORCH_JIT"] = "0"

    lazy_config: _config_t = args.config

    # Before creating the engine across processes, check if the config file exists and recover the engine if it does.
    lazy_engine = lazy_config.engine
    lazy_engine.params = _lazy.instantiate(lazy_engine.params)
    config_path = file_io.Path(lazy_engine.params.root) / "config.yaml"
    do_recover = config_path.exists()

    state = accelerate.PartialState()
    state.wait_for_everyone()  # Ensure the config file is not created before all processes validate its existence

    engine: up.engine.Engine = _lazy.instantiate(lazy_config.engine)

    if do_recover:
        _logger.info(
            "A serialized config YAML file exists at %s. Recovering engine at latest checkpoint.",
            config_path,
        )
        engine.recover()
    elif state.is_main_process:
        _logger.info("Storing serialized config to YAML file %s", engine.path)
        LazyConfig.save(lazy_config, str(config_path))

    # Setup dataloaders
    loaders: dict[str, up.data.DataLoaderFactory] = _lazy.instantiate(args.config.data.loaders)

    model_factory = ModelFactory(lazy_config.model, args.weights or None)

    if args.evaluation:
        results = engine.evaluate(model_factory, loaders[args.dataloader_test])
        _logger.info("Evaluation results: \n%s", create_table(results, format="long"))
    else:
        engine.train(
            model_factory,
            loaders[args.dataloader_train],
            trial=None,
            evaluation_loader_factory=loaders[args.dataloader_test],
        )


if __name__ == "__main__":
    command.root("train")
