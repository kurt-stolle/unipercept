r"""
CLI utility functions.
"""

from __future__ import annotations

import abc
import argparse
import inspect
import sys
import typing as T
from hashlib import md5


class _SubcommandTemplate(metaclass=abc.ABCMeta):
    """
    Straightforward subcommand template class.

    Each subcommand must define a `setup` and `main` method.
    """

    __slots__ = ()

    registry: dict[str, type[T.Self]] | None = None

    def __init_subclass__(cls, *, name: str | None = None):
        if cls.registry is None:
            msg = "Subcommand registry must be defined before defining subcommands."
            raise TypeError(msg)
        if name is not None:
            cls.registry[name] = cls

    @staticmethod
    @abc.abstractmethod
    def setup(prs: argparse.ArgumentParser):  # noqa: U100
        ...

    @staticmethod
    @abc.abstractmethod
    def main(args: argparse.Namespace):  # noqa: U100
        ...

    @classmethod
    def apply(cls, prs: argparse.ArgumentParser):
        if cls.registry is None:
            msg = "Subcommand registry must be defined before applying subcommands."
            raise TypeError(msg)
        handlers: dict[str, T.Callable[[argparse.Namespace], None]] = {}
        cmd = prs.add_subparsers(dest="subcommand", required=True)
        for name, sub in cls.registry.items():
            doc = next(
                (
                    search_doc
                    for cursor in (sub, sub.main, sub.setup)
                    if (search_doc := getattr(sub, "__doc__", None)) is not None
                    and len(search_doc) > 0
                ),
                f"run the {name} subcommand",
            )
            doc.strip()
            subprs = cmd.add_parser(name, help=doc)
            sub.setup(subprs)
            handlers[name] = sub.main

        return handlers

    def __new__(cls, prs: argparse.ArgumentParser):
        handlers = cls.apply(prs)

        def closure(args):
            cmd = handlers.get(args.subcommand)
            if cmd is None:
                print(
                    f"Unknown subcommand: {args.datasets_subcommand}", file=sys.stderr
                )
                sys.exit(1)
            else:
                cmd(args)

        return closure


def create_subtemplate() -> type[_SubcommandTemplate]:
    name = str(inspect.stack())
    return type(
        md5(name.encode("utf8"), usedforsecurity=False).hexdigest(),
        (_SubcommandTemplate,),
        {"registry": {}},
    )
