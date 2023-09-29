from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch.nn as nn
import torch.utils.data
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.file_io import PathManager

from ._cmd import command, logger
from ._utils import load_dataset, load_model, prompt_confirm, setup_config

__all__ = []


def _load_checkpoint_(path: str, model: nn.Module) -> None:
    if not PathManager.exists(path):
        raise ValueError(f"Checkpoint file does not exist: {path}")

    DetectionCheckpointer(model).load(path)


def _setup_output_dir(path: Path, reset: bool, resume: bool) -> Path:
    logger.info("setting up output directory: %s", path.as_posix())

    path = path.expanduser().resolve()
    if path.exists():
        if reset:
            shutil.rmtree(path)
        elif not resume:
            raise ValueError(f"Output directory already exists: {path}")

    path.mkdir(parents=True, exist_ok=resume)

    return path


def main(args: argparse.Namespace):
    if args.headless:
        tqdm = lambda x, *a, **kw: x
    else:
        from tqdm import tqdm

    prompt_confirm(
        "You have selected the potentially destructive flag `--reset`. This will remove the output directory.",
        args.reset and not args.headless,
    )

    output_dir = _setup_output_dir(args.output, args.reset, args.resume)
    cfg = setup_config(args.config)
    model = load_model(cfg.model, args.device).eval()
    _load_checkpoint_(args.checkpoint, model)

    dataset = load_dataset(cfg.dataloader.test)

    def get_output_path(in_) -> Path:
        output_path = output_dir
        output_file = "{id}.pt"
        if args.format == "sequence_id":
            assert "sequence_id" in in_, list(in_.keys())

            output_path = output_path / in_["sequence_id"]
            output_path.mkdir(exist_ok=True)
            output_path = output_path / output_file.format(id=in_["frame"])
        elif args.format == "image_id":
            output_path = output_path / output_file.format(id=in_["image_id"])
        else:
            raise ValueError(f"Unknown output format: {args.format}")

        return output_path

    for inputs in tqdm(dataset):
        # Remove already processed items if resuming
        if args.resume:
            in_remove = set()
            for in_idx, in_ in enumerate(inputs):
                if get_output_path(in_).exists():
                    in_remove.add(in_idx)

            for in_idx in in_remove:
                inputs.pop(in_idx)

        # Run model
        outputs = model(inputs)
        for in_, out in zip(inputs, outputs):
            output_path = get_output_path(in_)
            torch.save(out, output_path)


@command(help="run inference on items in a directory that are structured as a dataset")
@command.with_config
def infer(subparser: argparse.ArgumentParser):
    subparser.add_argument("--headless", action="store_true", help="disable all interactive prompts")
    subparser.add_argument("--device", type=str, default="cuda", help="device to use for inference")
    subparser.add_argument("--checkpoint", type=str, required=True, help="path to the checkpoint file")

    options_resume = subparser.add_mutually_exclusive_group(required=False)
    options_resume.add_argument(
        "--reset",
        action="store_true",
        help="whether to reset the output directory and caches if they exist",
    )
    options_resume.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume inference, skipping items that have already been processed",
    )

    options_io = subparser.add_argument_group("io")
    options_io.add_argument(
        "--format",
        type=str,
        default="image_id",
        help="format of the output file name, defaults to `image_id`",
        choices=["image_id", "sequence_id"],
    )
    options_io.add_argument("--output", "-o", type=Path, required=True, help="output directory")

    return main
