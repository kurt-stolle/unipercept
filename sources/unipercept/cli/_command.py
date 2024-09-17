from __future__ import annotations

import argparse
import functools
import typing as T
from collections.abc import Callable
from pathlib import Path
from typing import Concatenate, Generic, ParamSpec, TypeAlias

import unipercept.log

__all__ = ["command", "logger"]

logger = unipercept.log.get_logger()

MainParams = ParamSpec("MainParams")
Main = Callable[Concatenate[argparse.Namespace, MainParams], None]
CommandParams = ParamSpec("CommandParams")
Command = Callable[Concatenate[argparse.ArgumentParser, CommandParams], Main]
CommandExtension: TypeAlias = Callable[[Command], Command]


class command(Generic[CommandParams]):
    _registry = {}
    _signatures = {}

    def __init__(self, name: str | None = None, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self, func: Command) -> Command:
        name = self.name or str(func.__name__).replace("_", "-")
        if name in self._registry:
            if self._signatures[name] != hash(func):
                msg = f"Command {name} already registered! {self._registry[name][0].__module__}"
                raise ValueError(msg)
            return func

        self._registry[name] = (func, self.kwargs)
        self._signatures[name] = hash(func)
        return func

    @staticmethod
    def with_config(func: Command, **kwargs_add) -> Command:
        """Decorator for commands that require a configuration file."""

        from ._config import add_config_args

        @functools.wraps(func)
        def wrapper(parser, *args, **kwargs):
            add_config_args(parser, **kwargs_add)
            return func(parser, *args, **kwargs)

        return wrapper

    @classmethod
    def get_parser(cls) -> argparse.ArgumentParser:
        """Create the parser of the root CLI command."""

        parser = argparse.ArgumentParser(
            prog="unipercept",
            description="Unified Perception CLI for configuration-based training, evaluation and research.",
            epilog="See `unipercept <command> -h` for more information on a specific command.",
        )
        parser.add_argument(
            "--version", action="version", version=f"%(prog)s {unipercept.__version__}"
        )
        parser_cmd = parser.add_subparsers(title="command", required=True)

        # Register commands
        for name, (func, kwargs) in cls._registry.items():
            subparser = parser_cmd.add_parser(name, **kwargs)
            subparser.set_defaults(func=func(subparser))

        return parser

    @classmethod
    def root(cls, name: str | None = None, *args, **kwargs) -> None:
        """Parse arguments and call the appropriate command."""
        import sys

        import matplotlib

        matplotlib.use("Agg")

        if name is not None:
            if name.endswith(".py"):
                name = Path(name).stem
            sys.argv.insert(1, name)

        parser = cls.get_parser()
        args = parser.parse_args(*args, **kwargs)
        args.func(args)


def prompt_confirm(
    message: str, condition: bool, default: bool = False
) -> None | T.NoReturn:
    """
    Prompt the user for confirmation when a potentially destructive action is
    about to be performed.

    Parameters
    ----------
    message
        The message to display to the user, concatenated with "Are you sure? [y/N]".
    condition
        The condition that triggers the prompt (e.g. `True` if the action is about to be performed).
    default
        The default choice if the user does not provide any input. Defaults to `False`, i.e. "No".

    Returns
    -------
    Exits if the user aborts the action, otherwise None.
    """
    if not condition:
        return

    if default:
        message = f"{message} [Y/n] "
    else:
        message = f"{message} [y/N] "

    def input_choice() -> bool:
        while True:
            choice = input(message).lower()

            if choice in {"y", "yes"}:
                return True
            if choice in {"n", "no"}:
                return False
            if choice == "":
                return default

    choice = input_choice()
    if not choice:
        print("Aborting.")
        exit(0)
