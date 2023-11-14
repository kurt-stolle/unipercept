"""
FIXME: Does not work currently. Not used in the codebase. Consider removing.
"""

from __future__ import annotations

import argparse

import unicore.catalog
from typing_extensions import override

from unipercept.utils.function import to_sequence

__all__ = ["add_metadata_args"]


class MetadataFromNamesAction(argparse.Action):
    """
    Action to load a metadata dict from the command line, e.g. `--meta-names cityscapes coco`.
    Metadata is merged in order of appearance.
    """

    @override
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            parser.exit(message="No metadata names specified!\n", status=1)

        meta_old = getattr(namespace, self.dest, {})
        meta_new = unicore.catalog.get_info(*to_sequence(values))

        meta = {**meta_old, **meta_new}

        setattr(namespace, self.dest, meta)


class MetadataSetValueAction(argparse.Action):
    """
    Action to set a metadata variable from the command line, e.g. `--meta-set a.b.c 1`.
    Lists and tuples are supported, e.g. `--meta-set a.0.b 1` -> `a[0].b = 1`.
    """

    @override
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            parser.exit(message="No metadata key-value pair specified!\n", status=1)

        key, value = values

        meta = getattr(namespace, self.dest, {})
        self.walk_and_set_(meta, key, value)

        setattr(namespace, self.dest, meta)

    @staticmethod
    def walk_and_set_(meta, key, value):
        parts = key.split(".")
        current = meta
        for part in parts[:-1]:
            try:
                current = current[part]
            except KeyError as err:
                try:
                    current = current[int(part)]
                except ValueError:
                    raise err

        part = parts[-1]
        try:
            current[part] = value
        except KeyError as err:
            try:
                current[int(part)] = value
            except ValueError:
                raise err


def add_metadata_args(parser: argparse.ArgumentParser, *, flags=("--meta",), required=False, dest="metadata") -> None:
    """Adds metadata options to the argument parser."""

    meta_group = parser.add_argument_group("metadata")
    meta_group.add_argument(
        *map(lambda f: f + "-names", flags),
        dest=dest,
        type=str,
        action=MetadataFromNamesAction,
        nargs="+",
        required=required,
        metavar="NAME",
        help="load a metadata from registered dataset names",
    )
    meta_group.add_argument(
        *map(lambda f: f + "-set", flags),
        dest=dest,
        type=str,
        metavar=("K", "V"),
        action=MetadataSetValueAction,
        help="set a metadata variable `K` to value `V` where `K` is a dot-separated path, e.g. `a.b.0=1`",
        nargs=2,
    )
