"""Simple CLI extension for loading configuration files."""

from __future__ import annotations

import argparse
import typing as T
from pathlib import Path

from bullet import Bullet, Input
from omegaconf import DictConfig, OmegaConf
from typing_extensions import override

__all__ = ["add_config_args"]

NONINTERACTIVE_MODE = False


class ConfigLoad(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        from hydra.core.override_parser.overrides_parser import OverridesParser

        assert "type" not in kwargs, "Cannot specify type for ConfigLoad action!"

        super().__init__(option_strings, dest, type=str, **kwargs)

        self.overrides_parser = OverridesParser.create()

    @override
    def __call__(self, parser, namespace, values, option_string=None):
        from detectron2.config import LazyConfig

        if values is None or len(values) == 0:
            if NONINTERACTIVE_MODE:
                parser.exit(message="No configuration file specified!\n", status=1)
                return
            values = self.interactive()

        name, *overrides = values

        cfg = LazyConfig.load(name)
        cfg = self.apply_overrides(cfg, overrides)

        setattr(namespace, self.dest, cfg)

    def interactive(self) -> list[str]:
        """Interactively build the ``--config`` arguments."""
        values = [self.interactive_select(), *self.interactive_override()]
        return values

    @staticmethod
    def interactive_select(configs_root="./configs") -> str:
        print(
            f"\nNo configuration file specified (--config <path> [config.key=value ...]). Searching for configuration files in {configs_root}..."
        )

        root = Path(configs_root).resolve()
        choices = [str(p.relative_to(root)) for p in root.glob("**/*.py") if p.is_file() and not p.name.startswith("_")]
        try:
            action = Bullet(prompt="Select a configuration file:", choices=choices, bullet=" >", margin=2, pad_right=2)  # type: ignore
        except KeyboardInterrupt:
            print("No configuration file selected. Exiting.")
            exit(1)
        choice = action.launch()  # type: ignore
        choice = str(root / choice)

        print(f"Using configuration file: {choice}\n")

        return choice

    @staticmethod
    def interactive_override() -> T.Iterable[str]:
        """Provide overrides interactively."""
        return []

        # print("You can override any configuration value interactively. For example, a configuration 'model.device' could be overriden to 'gpu' by entering 'model.device=gpu'. Enter empty value to finish.")

        # while True:
        #     try:
        #         action = Input(prompt="Enter override: ")
        #     except KeyboardInterrupt:
        #         break
        #     choice = action.launch() # type: ignore
        #     if choice is None or len(choice) == 0:
        #         break
        #     yield choice

        # print("Configuration completed.\n")

    @staticmethod
    def safe_update(cfg, key, value):
        from omegaconf import OmegaConf

        parts = key.split(".")
        for idx in range(1, len(parts)):
            prefix = ".".join(parts[:idx])
            v = OmegaConf.select(cfg, prefix, default=None)
            if v is None:
                break
            if not OmegaConf.is_config(v):
                raise KeyError(f"Trying to update key {key}, but {prefix} is not a config, but has type {type(v)}.")
        OmegaConf.update(cfg, key, value, merge=True)

    def apply_overrides(self, cfg, overrides):
        for override in self.overrides_parser.parse_overrides(overrides):
            key = override.key_or_group
            value = override.value()
            self.safe_update(cfg, key, value)
        return cfg


def add_config_args(
    parser: argparse.ArgumentParser, *, flags=("--config",), required=True, nargs="*", **kwargs_add_argument
) -> None:
    """Adds a configuration file group to the argument parser."""
    assert all(f.startswith("-") for f in flags), "Flags must start with `-`!"

    group_config = parser.add_argument_group("configuration")
    group_config.add_argument(
        *flags,
        action=ConfigLoad,
        metavar=("FILE.py", "K=V"),
        nargs=nargs,
        required=required,
        help=(
            "path to the LazyConf file followed by key-value pairs to be merged into the config (in Hydra override "
            "notation, e.g. `--config a.b=c`)"
        ),
        **kwargs_add_argument,
    )
