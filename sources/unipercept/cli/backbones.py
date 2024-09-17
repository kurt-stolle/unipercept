"""
Commands that describe datasets.
"""

from __future__ import annotations

import argparse
import typing as T
from pprint import pformat

import pandas as pd
import regex as re

from unipercept.cli._command import command
from unipercept.log import create_table
from unipercept.nn import backbones
from unipercept.utils.cli import create_subtemplate
from unipercept.utils.seed import set_seed

__all__ = []


Subcommand = create_subtemplate()

command(
    name="backbones",
    help="inspect feature extraction networks from various pretrained sources",
)(Subcommand)
set_seed()


def _add_source_name_args(prs: argparse.ArgumentParser):
    prs.add_argument("source", type=str, help="source of the backbone")
    prs.add_argument("name", type=str, help="name of the backbone")


class ListSubcommand(Subcommand, name="list"):
    """
    Lists all available backbones, optionally filtering by source and pretrained status.
    """

    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument(
            "--pretrained",
            "-p",
            action="store_true",
            help="list only pretrained backbones",
        )
        prs.add_argument(
            "--match",
            "-m",
            default=None,
            type=re.compile,
            help="regex pattern to filter backbones by",
        )
        prs.add_argument(
            "--output",
            "-o",
            default=None,
            help="output file to save the discovered backbones to (CSV format)",
        )
        prs.add_argument(
            "sources",
            default=["torchvision", "timm"],
            nargs="*",
            help="framework to list backbones of, choices: [torchvision, timm]",
        )

    @staticmethod
    def read_backbones(name: str, pretrained: bool):
        match name:
            case "torchvision":
                yield from backbones.torchvision.list_available(pretrained=pretrained)
            case "timm":
                yield from backbones.timm.list_available(pretrained=pretrained)
            case _:
                raise ValueError(f"Unknown source '{name}'")

    @classmethod
    def main(cls, args: argparse.Namespace):
        records = []
        for source in args.sources:
            for name in cls.read_backbones(source, args.pretrained):
                item = {"source": source, "name": name}
                if args.match is not None:
                    match = args.match.search(name)
                    if match is None:
                        continue
                    item["match"] = match.group()
                records.append(item)

        data = pd.DataFrame(records)

        print(create_table(data))

        if args.output:
            assert args.output.endswith(".csv"), "Output must be a CSV file"
            data.to_csv(args.output, index=False)


class EchoSubcommand(Subcommand, name="echo"):
    """
    Shows the module structure of a backbone by directly outputting its representation
    to stdout.
    """

    @staticmethod
    def setup(prs: argparse.ArgumentParser):
        _add_source_name_args(prs)

    @classmethod
    def main(cls, args: argparse.Namespace):
        backbone = backbones.catalog.get(args.source)(name=args.name)
        print(backbone)


class InfoSubcommand(Subcommand, name="info"):
    """
    Shows the feature information object of a backbone.
    """

    @staticmethod
    def setup(prs: argparse.ArgumentParser):
        _add_source_name_args(prs)

    @classmethod
    def main(cls, args: argparse.Namespace):
        backbone = backbones.catalog.get(args.source)(name=args.name)
        print(pformat(backbone.feature_info))


class NodesSubcommand(Subcommand, name="nodes"):
    """
    Shows the feature information object of a backbone.
    """

    @staticmethod
    def setup(prs: argparse.ArgumentParser):
        _add_source_name_args(prs)

    @classmethod
    def main(cls, args: argparse.Namespace):
        nodes_train, nodes_eval = backbones.catalog.get(args.source).list_nodes(
            name=args.name
        )

        if set(nodes_train) != set(nodes_eval):
            nodes = {"training": nodes_train, "inference": nodes_eval}
        else:
            nodes = nodes_train

        print(pformat(nodes))


if __name__ == "__main__":
    command.root("backbones")
