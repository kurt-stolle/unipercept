"""
Commands that describe datasets.
"""

from __future__ import annotations

import argparse
import dataclasses as D
import sys
import typing as T

import pandas as pd
from tabulate import tabulate_formats

import unipercept as up
from unipercept.cli._command import command, logger

__all__ = []


@command(help="dataset operations")
def datasets(prs: argparse.ArgumentParser):
    """
    Defines the following subcommands:
    - ``ls`` - list available datasets
    - ``info`` - show information about a dataset, also known as 'metadata'.
    - ``manifest`` - show the manifest of a dataset
    """

    from tabulate import tabulate_formats

    subprs = prs.add_subparsers(dest="datasets_subcommand", required=True)
    subprs.add_parser("list", help="list available datasets")

    variants = subprs.add_parser(
        "list-variants", help="list available dataset variants"
    )
    variants.add_argument("dataset", nargs="+", help="dataset name(s)")

    stats = subprs.add_parser(
        "stats", help="show statistics about the subvariants of a dataset"
    )
    stats.add_argument(
        "--output",
        default=None,
        type=up.file_io.Path,
        help="path to save the output dataframe to",
    )
    stats.add_argument("dataset", nargs="+", help="dataset name(s)")

    info = subprs.add_parser("info", help="show information about a dataset")
    info.add_argument(
        "--format",
        type=str,
        default="simple",
        help="output format, options: " + ", ".join(tabulate_formats),
    )
    info.add_argument("dataset", help="dataset name")

    return _main


def _main(args):
    match args.datasets_subcommand:
        case "list":
            _main_ls()
        case "list-variants":
            _main_ls_variants(args)
        case "stats":
            _main_stats(args)
        case "info":
            _main_info(args)
        case _:
            msg = f"Unknown subcommand: {args.datasets_subcommand}"
            raise ValueError(msg)


def _main_ls():
    """
    List available datasets.
    """
    for ds in up.data.sets.catalog.list_datasets():
        print(ds)


def _main_ls_variants(args):
    """
    List available dataset variants.
    """
    df_list: list[pd.DataFrame] = []
    for ds_key in args.dataset:
        ds_cls = up.data.sets.catalog.get_dataset(ds_key)
        df = pd.DataFrame(ds_cls.variants())
        df["dataset"] = args.dataset

        df_list.append(df)

    df_all = pd.concat(df_list)

    print(up.log.create_table(df_all))


def _main_stats(args):
    st_list: list[dict[str, T.Any]] = []
    for ds_key in args.dataset:
        ds_cls = up.data.sets.catalog.get_dataset(ds_key)
        for variant in ds_cls.variants():
            print(variant)
            ds = ds_cls(**variant)

            queue_size = len(ds.queue)

            st_list.append(
                {
                    "dataset": ds_key,
                    **variant,
                    "queue_size": queue_size,
                }
            )
    st = pd.DataFrame(st_list)
    print(up.log.create_table(st))

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


def _main_info(args):
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


if __name__ == "__main__":
    command.root("datasets")
