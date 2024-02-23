"""Simple CLI extension for loading configuration files."""

from __future__ import annotations

import argparse
import enum
import typing as T

from bullet import Bullet
from omegaconf import DictConfig
from typing_extensions import override

import unipercept as up
from unipercept import file_io

__all__ = ["add_config_args", "ConfigFileContentType"]

_NONINTERACTIVE_MODE = False
_BULLET_STYLES = {"bullet": " >", "margin": 2, "pad_right": 2}

up.list_datasets()  # trigger dataset registration

if T.TYPE_CHECKING:

    class ConfigFileContentType(DictConfig):
        @property
        def ENGINE(self) -> DictConfig:
            return self.engine

        @property
        def MODEL(self) -> DictConfig:
            return self.model

else:
    ConfigFileContentType: T.TypeAlias = DictConfig


class ConfigSource(enum.StrEnum):
    TEMPLATES = enum.auto()
    CHECKPOINTS = enum.auto()


class ConfigLoad(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        from hydra.core.override_parser.overrides_parser import OverridesParser

        assert "type" not in kwargs, "Cannot specify type for ConfigLoad action!"

        super().__init__(option_strings, dest, type=str, **kwargs)

        self.overrides_parser = OverridesParser.create()

    @override
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None or len(values) == 0:
            if _NONINTERACTIVE_MODE:
                parser.exit(message="No configuration file specified!\n", status=1)
                return
            values = self.interactive()

        name, *overrides = values

        cfg = up.read_config(name)
        cfg = self.apply_overrides(cfg, overrides)
        cfg["_CLI"] = name

        setattr(namespace, self.dest, cfg)

    def interactive(self) -> list[str]:
        """Interactively build the ``--config`` arguments."""
        values = [self.interactive_select(), *self.interactive_override()]
        return values

    @staticmethod
    def interactive_select(configs_root="//configs/") -> str:
        print(
            "No configuration file specified (--config <path> [config.key=value ...])."
        )

        # Prompt 1: Where to look for configurations?
        try:
            action = Bullet(prompt="Select a configuration source:", choices=[v.value for v in ConfigSource], **_BULLET_STYLES)  # type: ignore
        except KeyboardInterrupt:
            print("Received interrupt singal. Exiting.")
            exit(1)
            return
        choice = action.launch()  # type: ignore
        choice = ConfigSource(choice)

        match choice:
            case ConfigSource.TEMPLATES:
                configs_root = file_io.Path("//configs/")
            case ConfigSource.CHECKPOINTS:
                configs_root = file_io.Path("//output/")
            case _:
                msg = f"Invalid choice: {action}"
                raise ValueError(msg)

        configs_root = configs_root.expanduser().resolve()
        config_candidates = configs_root.glob("**/*")
        config_candidates = list(
            filter(
                lambda f: f.is_file()
                and not f.name.startswith("_")
                and f.suffix in (".py", ".yaml"),
                config_candidates,
            )
        )
        config_candidates.sort()

        if len(config_candidates) == 0:
            print(f"No configuration files found in {str(configs_root)}.")
            exit(1)
            return
        else:
            print(
                f"Found {len(config_candidates)} configuration files in {str(configs_root)}."
            )

        # Prompt 2: Which configuration file to use?
        choices = [str(p.relative_to(configs_root)) for p in config_candidates]
        try:
            action = Bullet(prompt="Select a configuration file:", choices=choices, **_BULLET_STYLES)  # type: ignore
        except KeyboardInterrupt:
            print("Received interrupt singal. Exiting.")
            exit(1)
            return
        choice = action.launch()  # type: ignore
        choice = str(configs_root / choice)

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
                raise KeyError(
                    f"Trying to update key {key}, but {prefix} is not a config, but has type {type(v)}."
                )
        OmegaConf.update(cfg, key, value, merge=True)

    def apply_overrides(self, cfg, overrides):
        for override in self.overrides_parser.parse_overrides(overrides):
            key = override.key_or_group
            value = override.value()
            self.safe_update(cfg, key, value)
        return cfg


def add_config_args(
    parser: argparse.ArgumentParser,
    *,
    flags=("--config",),
    required=True,
    nargs="*",
    **kwargs_add_argument,
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
