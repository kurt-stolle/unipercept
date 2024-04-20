"""
Commands that describe datasets.
"""

from __future__ import annotations

import abc
import argparse
import dataclasses as D
import sys
import typing as T

import pandas as pd
import typing_extensions as TX
from tabulate import tabulate_formats
from tqdm import tqdm

from unipercept import file_io
from unipercept.cli._command import command, logger
from unipercept.log import create_table
from unipercept.utils.string import to_snake_case

__all__ = []


class Subcommand(metaclass=abc.ABCMeta):
    """
    Quick and dirty subcommand implementation.
    """

    __slots__ = ()

    registry: T.ClassVar[dict[str, type[T.Self]]] = {}

    def __init_subclass__(cls, *, name: str):
        if name is None:
            name = to_snake_case(cls.__name__)
        cls.registry[name] = cls

    def __new__(cls):
        msg = f"Subcommand {self.__name__} must not be initialized."
        raise TypeError(msg)

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
        handlers: dict[str, T.Callable[[argparse.Namespace], None]] = {}
        cmd = prs.add_subparsers(dest="subcommand", required=True)
        for name, sub in cls.registry.items():
            doc = sub.__call__.__doc__
            if doc is None:
                doc = f"run the {name} subcommand"
            else:
                doc = doc.strip()
            subprs = cmd.add_parser(name, help=doc)
            sub.setup(subprs)
            handlers[name] = sub.main

        return handlers


@command(help="dataset operations")
def datasets(prs: argparse.ArgumentParser):
    handlers = Subcommand.apply(prs)

    def main(args):
        cmd = handlers.get(args.subcommand)
        if cmd is None:
            print(f"Unknown subcommand: {args.datasets_subcommand}", file=sys.stderr)
            sys.exit(1)
        else:
            cmd(args)

    return main


class ListSubcommand(Subcommand, name="list"):
    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        pass

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        """
        List available datasets.
        """
        from unipercept.data.sets import catalog

        for ds in catalog.list_datasets():
            print(ds)


class ListVariantsSubcommand(Subcommand, name="list-variants"):
    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument("dataset", nargs="+", help="dataset name(s)")

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        """
        List available dataset variants.
        """

        from unipercept.data.sets import catalog

        df_list: list[pd.DataFrame] = []
        for ds_key in args.dataset:
            ds_cls = catalog.get_dataset(ds_key)
            df = pd.DataFrame(ds_cls.variants())
            df["dataset"] = ds_key

            df_list.append(df)

        df_all = pd.concat(df_list)

        print(create_table(df_all))


class StatsSubcommand(Subcommand, name="stats"):
    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument("dataset", nargs="+", help="dataset name(s)")
        prs.add_argument(
            "--output",
            default=None,
            type=file_io.Path,
            help="path to save the output dataframe to",
        )

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        from unipercept.data.sets import catalog
        from unipercept.data.types import Manifest

        st_list: list[dict[str, T.Any]] = []
        for ds_key in args.dataset:
            ds_cls = catalog.get_dataset(ds_key)
            for variant in ds_cls.variants():
                ds = ds_cls(**variant)

                cap_count = 0
                img_count = 0
                dep_count = 0
                pan_count = 0

                mfst: Manifest = ds.manifest

                for seq_count, seq in enumerate(mfst["sequences"].values()):
                    seq_count += 1
                    caps = seq["captures"]
                    cap_count += len(caps)
                    for cap in caps:
                        src = cap["sources"]
                        if "image" in src:
                            img_count += 1
                        if "depth" in src:
                            dep_count += 1
                        if "panoptic" in src:
                            pan_count += 1

                st_list.append(
                    {
                        "dataset": ds_key,
                        **variant,
                        "sequences": seq_count,
                        "captures": cap_count,
                        "images": img_count,
                        "depths": dep_count,
                        "panoptics": pan_count,
                    }
                )
        st = pd.DataFrame(st_list)
        print(create_table(st, format="wide"))

        out_file = args.output
        if out_file is None:
            return

        match out_file.suffix:
            case ".csv":
                st.to_csv(out_file, index=False)
            case ".xlsx":
                st.to_excel(out_file, index=False)
            case _:
                msg = f"Unsupported file format: {out_file.suffix}"
                raise ValueError(msg)


class InfoSubcommand(Subcommand, name="info"):
    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument(
            "--format",
            type=str,
            default="simple",
            help="output format, options: " + ", ".join(tabulate_formats),
        )
        prs.add_argument("dataset", help="dataset name")

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        """
        Show information about a dataset, also known as 'metadata'.
        """
        from tabulate import tabulate

        from unipercept import get_info

        info = get_info(args.dataset)
        rows = []
        for key, value in sorted(D.asdict(info).items(), key=lambda x: x[0]):
            if isinstance(value, T.Mapping):
                rows.append((key, ""))  # type(value).__name__))
                for k, v in value.items():
                    rows.append((f"[ {k!r} ]", v))
            elif isinstance(value, T.Iterable) and not isinstance(value, str):
                rows.append((key, ""))  # type(value).__name__))
                for i, v in enumerate(value):
                    rows.append((f"[ {i!r} ]", v))
            else:
                rows.append((key, value))

        table = tabulate(rows, tablefmt=args.format)
        print(table, flush=True, file=sys.stdout)


class CacheSubcommand(Subcommand, name="cache"):
    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument(
            "--purge", action="store_true", help="purge the cache (if it exists)"
        )
        prs.add_argument("dataset", help="dataset name")

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        """
        Cache the dataset.
        """
        from unipercept.data.sets import catalog

        ds_cls = catalog.get_dataset(args.dataset)

        stats = []
        for variant in ds_cls.variants():
            ds = ds_cls(**variant)
            path_str = ds.cache_path
            path = file_io.Path(path_str)

            exists = path.is_file()
            if args.purge and exists:
                path.unlink()

            stats.append(
                {
                    "dataset": args.dataset,
                    **variant,
                    "cached": exists,
                    "path": path_str,
                }
            )

        df = pd.DataFrame(stats)
        print(create_table(df, format="wide"))


if __name__ == "__main__":
    command.root("datasets")
