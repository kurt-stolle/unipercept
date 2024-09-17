"""Simple CLI extension for loading configuration files."""

from __future__ import annotations

import argparse
import enum
import os
import typing as T
from typing import override

import pandas as pd
import torch
from bullet import Bullet
from omegaconf import DictConfig

from unipercept import file_io
from unipercept.log import create_table, logger

__all__ = ["add_config_args", "ConfigFileContentType"]

NONINTERACTIVE: T.Final = False
BULLET_STYLES: T.Final = {"bullet": " >", "margin": 2, "pad_right": 2}
ARGPARSE_DEST_CONFIG: T.Final = "config"

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
        from unipercept import read_config

        if values is None or len(values) == 0:
            if NONINTERACTIVE:
                parser.exit(message="No configuration file specified!\n", status=1)
                return
            values = self.interactive()

        name, *overrides = values

        cfg = read_config(name)
        cfg = self.apply_overrides(cfg, overrides)
        cfg["CLI"] = name
        cfg["CLI_OVERRIDES"] = list(overrides)

        setattr(namespace, self.dest + "_path", name)
        setattr(namespace, self.dest + "_overrides", overrides)
        setattr(namespace, self.dest, cfg)

    def interactive(self) -> list[str]:
        """Interactively build the ``--config`` arguments."""
        values = [self.interactive_select(), *self.interactive_override()]
        return values

    @staticmethod
    def interactive_select(configs_root="configs://") -> str:
        print(
            "No configuration file specified (--config <path> [config.key=value ...])."
        )

        # Prompt 1: Where to look for configurations?
        try:
            action = Bullet(
                prompt="Select a configuration source:",
                choices=[v.value for v in ConfigSource],
                **BULLET_STYLES,
            )  # type: ignore
        except KeyboardInterrupt:
            print("Received interrupt singal. Exiting.")
            exit(1)
            return None
        choice = action.launch()  # type: ignore
        choice = ConfigSource(choice)

        match choice:
            case ConfigSource.TEMPLATES:
                configs_root = file_io.Path("configs://")
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
            return None
        print(
            f"Found {len(config_candidates)} configuration files in {str(configs_root)}."
        )

        # Prompt 2: Which configuration file to use?
        choices = [str(p.relative_to(configs_root)) for p in config_candidates]
        try:
            action = Bullet(
                prompt="Select a configuration file:", choices=choices, **BULLET_STYLES
            )  # type: ignore
        except KeyboardInterrupt:
            print("Received interrupt singal. Exiting.")
            exit(1)
            return None
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
                msg = f"Trying to update key {key}, but {prefix} ({type(v)}) is not configurable."
                raise KeyError(msg)
        OmegaConf.update(cfg, key, value, merge=True)

    def apply_overrides(self, cfg, overrides):
        overrides_applied = []
        for override in self.overrides_parser.parse_overrides(overrides):
            key = override.key_or_group
            value = override.value()
            if value == "None":
                value = None
            self.safe_update(cfg, key, value)
            overrides_applied.append(
                {"Key": key, "Value": value, "Type": type(value).__name__}
            )
        if len(overrides_applied) > 0:
            logger.info(
                "Configuration overrides applied from CLI:\n%s",
                create_table(
                    pd.DataFrame.from_records(overrides_applied), format="wide"
                ),
            )
        return cfg


class ConfigMode(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, type=None, nargs=0, **kwargs)

        self.key = kwargs.pop("key", ARGPARSE_DEST_CONFIG)

    @override
    def __call__(self, parser, namespace, values, option_string=None):
        cfg = getattr(namespace, self.key)

        if cfg is None:
            msg = "Cannot apply patch when configuration is not loaded!"
            raise RuntimeError(msg)

        self.apply_patch(cfg)

    def apply_patch(self, cfg: ConfigFileContentType):
        raise NotImplementedError()

    FLAGS: T.ClassVar[tuple[str] | None] = None
    HELP: T.ClassVar[str | None] = None

    def __init_subclass__(cls, *, flags: str | T.Iterable[str], help: str, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.FLAGS = (flags,) if isinstance(flags, str) else tuple(flags)
        cls.HELP = help

    @classmethod
    def apply_parser(cls, parser):
        if cls.FLAGS is None:
            msg = "No flags specified for the configuration patch!"
            raise ValueError(msg)
        parser.add_argument(
            *cls.FLAGS,
            action=cls,
            help=cls.HELP,
        )


class ConfigDebugMode(
    ConfigMode, flags="--debug", help="patches the configuration for model debugging"
):
    @override
    def apply_patch(self, cfg):
        from logging import DEBUG

        from unipercept.engine.debug import DebugMode

        logger.info("Applying debug mode to the configuration.")

        os.environ["TORCH_DEBUG"] = "1"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        os.environ["PYTORCH_DEBUG"] = "1"
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["UP_LOGS_LEVEL"] = "DEBUG"

        torch._logging.set_logs(dynamo=DEBUG)
        torch._dynamo.reset()
        torch._dynamo.config.verbose = True

        cfg.ENGINE.params.debug = DebugMode.UNDERFLOW_OVERFLOW


class ConfigDetectionAnomaliesMode(
    ConfigMode,
    flags="--detect-anamolies",
    help=(
        "Enable anamoly detection. "
        "Note that using this in FP16 mode will likely result in false positives "
        "due to grad scaler warmup."
    ),
):
    @override
    def apply_patch(self, cfg):
        torch.autograd.set_detect_anomaly(True)


class ConfigDeterministicMode(
    ConfigMode,
    flags="--deterministic",
    help="patches the configuration to enable deterministic mode",
):
    @override
    def apply_patch(self, cfg):
        cfg.ENGINE.params.full_determinism = True


class ConfigDisableTrackers(
    ConfigMode,
    flags="--disable-trackers",
    help="patches the configuration to disable all experiment trackers",
):
    @override
    def apply_patch(self, cfg):
        cfg.ENGINE.params.trackers = []


def add_config_args(
    parser: argparse.ArgumentParser,
    *,
    flags=("--config", "-C"),
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
        dest=ARGPARSE_DEST_CONFIG,
        metavar=("FILE.py", "K=V"),
        nargs=nargs,
        required=required,
        help=(
            "path to the LazyConf file followed by key-value pairs to be merged into the config (in Hydra override "
            "notation, e.g. `--config a.b=c`)"
        ),
        **kwargs_add_argument,
    )
    # group_config.add_argument(
    #    *flags_patch,
    #    action=PatchLoad,
    #    dest=ARGPARSE_DEST_PATCH,
    #    help="apply configuration patches",
    #    **kwargs_add_argument,
    # )

    ConfigDebugMode.apply_parser(parser)
    ConfigDisableTrackers.apply_parser(parser)
