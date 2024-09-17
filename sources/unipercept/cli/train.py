"""Training entry point."""

from __future__ import annotations

import argparse
import os
import sys

import unipercept as up
from unipercept.cli._command import command, logger
from unipercept.cli._config import ConfigFileContentType as config_t

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
        "--freeze-weights",
        "-F",
        action="store_true",
        default=False,
        help="freeze all weights that were loaded from the given `--weights` argument",
    )
    p.add_argument(
        "--resume",
        "-R",
        action="store_true",
        help="continue training from the last checkpoint",
    )
    p.add_argument(
        "--reduce-batch-size",
        type=int,
        default=1,
        help="factor by which to reduce all batch sizes and increase gradient accumulation steps",
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


def _step(args) -> config_t:
    if args.no_jit:
        logger.info("Disabling JIT compilation")
        os.environ["PYTORCH_JIT"] = "0"

    if args.reduce_batch_size > 1:
        power = int(2 ** (args.reduce_batch_size - 1))
        logger.info(
            "Reducing batch size by scale %d (batch_size / %d)",
            args.reduce_batch_size,
            power,
        )
        for stage in args.config.ENGINE.stages:
            stage.batch_size //= power
            if stage.batch_size < 1:
                raise ValueError("Batch size cannot be less than 1")
            if hasattr(stage, "gradient_accumulation"):
                stage.gradient_accumulation *= power
            else:
                stage.gradient_accumulation = power

    return args.config


def _main(args):
    lazy_config: config_t = _step(args)

    up.state.barrier()  # Ensure the config file is not created before all processes validate its existence

    engine: up.engine.Engine = up.config.lazy.instantiate(lazy_config.ENGINE)

    if args.resume:
        session_id_recovered = lazy_config[KEY_SESSION_ID]
        if session_id_recovered is None:
            msg = "Training configuration does not support resumption"
            raise ValueError(msg)
        config_path = up.file_io.Path(args.config_path)
        if not config_path.suffix.endswith(".yaml"):
            msg = "Cannot resume training without a YAML config file"
            raise ValueError(msg)
        resume_dir = config_path.parent
        if not (resume_dir / "outputs").exists():
            msg = "Cannot resume training from a directory without outputs"
            raise ValueError(msg)
        if not (resume_dir / "logs").exists():
            msg = "Cannot resume training from a directory without logs"
            raise ValueError(msg)

        engine.session_id = session_id_recovered
        engine.session_dir = resume_dir
    elif up.state.check_main_process():
        logger.info("Storing serialized config to YAML file %s", engine.config_path)
        lazy_config[KEY_SESSION_ID] = engine.session_id
        engine.config = lazy_config

    logger.info(
        "Starting engine session:\n%s",
        up.log.create_table(
            {
                "Session ID": engine.session_id,
                "Session path": str(engine.session_dir),
            }
        ),
    )

    # Setup dataloaders
    model_factory = up.model.ModelFactory(
        lazy_config.MODEL, weights=args.weights, freeze_weights=args.freeze_weights
    )
    try:
        if args.evaluation:
            logger.info(
                "Running in EVALUATION ONLY MODE. Be aware that no weights are loaded if not provided explicitly!"
            )
            results = engine.run_evaluation(model_factory)
            logger.info(
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

        logger.info(
            "Training interrupted by user. To resume, use the --resume flag and configuration file: %s",
            engine.config_path,
        )
        exit(0)


if __name__ == "__main__":
    command.root("train")
