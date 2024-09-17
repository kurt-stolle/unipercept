"""
Run a sanity check on a configuration file.
"""

from __future__ import annotations

import argparse
import functools
import typing as T

import torch
import typing_extensions as TX

import unipercept as up
from unipercept.cli._command import command, logger
from unipercept.engine.callbacks import CallbackDispatcher, Signal
from unipercept.log import create_table
from unipercept.model import ModelBase


@command(
    help="run sanity checks before spending budget on training or evaluation",
    description=__doc__,
)
@command.with_config
def check(p: argparse.ArgumentParser):
    p.add_argument(
        "--show-parameters", action="store_true", help="Show model parameters"
    )
    p.add_argument(
        "--show-forward", action="store_true", help="Show forward pass stats"
    )
    p.add_argument(
        "--show-backward", action="store_true", help="Show backward pass stats"
    )

    def _patch_config(args):
        config = args.config

        config.ENGINE.params.trackers = []
        config.ENGINE.callbacks.append(
            CheckCallback(
                show_parameters=args.show_parameters,
                show_forward=args.show_forward,
                show_backward=args.show_backward,
            )
        )
        for stage in config.ENGINE.stages:
            stage.batch_size = 2
        return config

    def _main(args):
        config = _patch_config(args)
        engine = up.create_engine(config)
        model_factory = up.create_model_factory(config)

        logger.info("Running TRAINING checks...")
        engine.run_training(model_factory)

        logger.info("Running INFERENCE checks...")
        print("(not implemented)")
        #    engine.run_inference(model, config)

    return _main


def register_hooks(model, show_forward: bool, show_backward: bool):
    def show_args(values: T.Iterable[T.Any]):
        out = []
        for i, x in enumerate(values):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                y = f"{x.shape} ({x.dtype})"

                stats = {}
                if x.numel() > 1:
                    stats["min"] = x.min().item()
                    stats["max"] = x.max().item()
                    try:
                        stats["mean"] = x.mean().item()
                        stats["std"] = x.std().item()
                        stats["median"] = x.median().item()
                        stats["norm"] = x.norm().item()
                    except Exception:
                        pass
                else:
                    stats["value"] = x.item()

                y += " " + ", ".join(f"{k}: {v:.2f}" for k, v in stats.items())
            else:
                y = str(x.__class__.__name__)

            out.append(f"[{i}] {y}")

        if len(out) == 0:
            return "(empty)"
        return "\n\t\t".join(out)

    def logging_hook(module, inputs, outputs, *, phase: str, name: str):
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]

        print(f"-- {phase}: {name} ({module.__class__.__name__}) --", flush=True)
        print(f"\tinputs: \n\t\t{show_args(inputs)}", flush=True)
        print(f"\toutputs: \n\t\t{show_args(outputs)}", flush=True)

    hooks = []
    for name, module in model.named_modules():
        if show_forward:
            hooks.append(
                module.register_forward_hook(
                    functools.partial(logging_hook, phase="forward", name=name)
                )
            )
        if show_backward:
            hooks.append(
                module.register_backward_hook(
                    functools.partial(logging_hook, phase="backward", name=name)
                )
            )
    return hooks


def human_readable_size(size: int) -> str:
    size = float(size)
    for unit in ["k", "M", "G", "T"]:
        size /= 1000
        if size < 1000:
            return f"{size:.1f} {unit}"


def show_parameters(model):
    table = {
        "name": [],
        "shape": [],
        "params": [],
        "dtype": [],
        "requires_grad": [],
    }
    n_params = 0
    n_params_grad = 0
    for name, param in model.named_parameters():
        numel = param.numel()
        table["name"].append(name)
        table["shape"].append(tuple(param.shape))
        table["params"].append(human_readable_size(numel))
        table["dtype"].append(param.dtype)
        table["requires_grad"].append(param.requires_grad)

        n_params += numel
        if param.requires_grad:
            n_params_grad += numel
    table_str = create_table(table, format="wide")
    logger.info(
        f"Model parameters (trainable {human_readable_size(n_params_grad)}/{human_readable_size(n_params)}):\n{table_str}"
    )


class CheckCallback(CallbackDispatcher):
    def __init__(
        self, *, show_parameters: bool, show_forward: bool, show_backward: bool
    ):
        self.hooks = []
        self.show_parameters = show_parameters
        self.show_forward = show_forward
        self.show_backward = show_backward

    @TX.override
    def on_train_step_begin(
        self, params, state, control: Signal, *, model: ModelBase, **kwargs
    ):
        logger.info("Registering check hooks...")
        hooks = register_hooks(model, self.show_forward, self.show_backward)
        self.hooks.extend(hooks)

        if self.show_parameters:
            show_parameters(model)

    @TX.override
    def on_train_step_end(self, params, state, control: Signal, **kwargs):
        logger.info("Removing check hooks...")
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        logger.info("Stopping training after one step for sanity check...")

        control.should_training_stop = True
        control.should_save = False
        control.should_log = False
        control.should_evaluate = False

        exit(0)


if __name__ == "__main__":
    command.root(__file__)
