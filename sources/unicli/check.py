"""Check a configuration file for errors by attempting to initialize the model, dataloader and optimizer."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy
import torch
import torch._dynamo
import torch._dynamo.config
import torch.nn as nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.engine import default_setup, default_writers, hooks, launch
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.utils import comm
from omegaconf import DictConfig
from pyparsing import Any

from unipercept.utils.time import get_timestamp

from ._cmd import command, logger
from ._utils import load_dataset, load_model, prompt_confirm, setup_config


def _make_model(cfg):
    logger.info("Initializing model...")
    return instantiate(cfg.model)


def main(args):
    cfg = args.config

    logger.info("Checking configuration file...")
    try:
        _make_model(cfg)
    except Exception as e:
        logger.error("Error while initializing model!")
        raise e

    logger.info("Configuration file is valid!")


@command(help="run inference on items in a directory that are structured as a dataset")
@command.with_config
def check(subparser: argparse.ArgumentParser):
    subparser.add_argument("--device", type=str, default="cuda", help="device to use for inference")

    return main
