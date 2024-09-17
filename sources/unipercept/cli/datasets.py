"""
Commands that describe datasets.
"""

from __future__ import annotations

import argparse
import dataclasses as D
import random
import sys
import typing as T

import pandas as pd
import torch
import typing_extensions as TX
from tabulate import tabulate_formats
from tqdm import tqdm

from unipercept import file_io, render
from unipercept.cli._command import command, logger
from unipercept.data.sets import PerceptionDataset
from unipercept.log import create_table
from unipercept.utils.cli import create_subtemplate
from unipercept.utils.inspect import generate_path
from unipercept.utils.seed import set_seed

__all__ = []


Subcommand = create_subtemplate()

command(name="datasets", help="dataset operations")(Subcommand)
set_seed()


def _add_datasets_arg(prs: argparse.ArgumentParser, nargs="+"):
    prs.add_argument("datasets", nargs=nargs, type=str, help="dataset name(s)")


def _add_variant_arg(prs: argparse.ArgumentParser):
    prs.add_argument("dataset", type=str, help="dataset name")
    prs.add_argument(
        "hash",
        type=str,
        help="variant hash (use 'list-variants' to show options) or keyword arguments",
    )


def _get_dataset_by_hash(args: argparse.Namespace) -> PerceptionDataset:
    # This is somewhat inefficient, but it's the easiest way to specify the dataset
    # without having to run unsafe code.
    from unipercept.data.sets import catalog

    hash_table = {}

    # Build hash table and quit prematurely if an exact match is found
    ds_cls = catalog.get(args.dataset)
    for variant in ds_cls.variants():
        ds = ds_cls(**variant)
        ds_hash = ds.hash
        if ds_hash == args.hash:
            return ds
        hash_table[ds_hash] = ds

    # Check whether the hash provided matches the first characters of any hash (short-hand)
    hash_len = len(args.hash)
    short_table = {k[:hash_len]: v for k, v in hash_table.items()}
    if len(short_table) != len(hash_table):
        msg = f"Hash is ambiguous: {args.dataset}({args.hash})"
        raise ValueError(msg)

    if args.hash in short_table:
        return short_table[args.hash]

    msg = f"Variant not found: {args.dataset}({args.hash})"
    raise ValueError(msg)


def _kwargs_to_str(variant: dict[str, T.Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in variant.items())


class ValidateSubcommand(Subcommand, name="validate"):
    """
    Check the dataset for integrity. This will check that the dataset has a
    valid manifest and that all files in the manifest are present/loadable.
    """

    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument(
            "--samples",
            "-n",
            type=int,
            default=-1,
            help="amount of samples to validate, -1 means all samples, 0 skips loading.",
        )
        _add_datasets_arg(prs)

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        from unipercept.data.sets import catalog

        for ds_key in args.datasets:
            ds_cls = catalog.get(ds_key)
            for variant in ds_cls.variants():
                logger.info("Validating %s", ds_key)

                ds_repr = f"{ds_key}({_kwargs_to_str(variant)})"
                ds = ds_cls(**variant)

                queue, pipe = ds()

                if (queue_len := len(queue)) != (pipe_len := len(pipe)):
                    logger.info(
                        "FAIL: queue (%d items) and pipe (%d items) lengths do not match",
                        queue_len,
                        pipe_len,
                    )
                    continue

                if args.samples == 0:
                    continue
                if args.samples == -1:
                    n = len(queue)
                elif args.samples > 0:
                    n = min(args.samples, len(queue))
                else:
                    msg = "Invalid sample size: %d" % args.samples
                    raise ValueError(msg)
                with tqdm(desc=ds_repr, total=n) as pbar:
                    for i in range(n):
                        d = pipe[i]
                        del d
                        pbar.update()


class ShowSubcommand(Subcommand, name="show"):
    """
    Show a sample from the dataset.
    """

    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument(
            "--index",
            "-i",
            type=str,
            nargs="+",
            default="?",
            help="Index in the dataset queue to show, where '?' shows a random sample (default)",
        )
        prs.add_argument(
            "--output",
            "-o",
            default=None,
            required=False,
            help="Path to save the output image to. If not provided, the image will be displayed.",
        )
        prs.add_argument(
            "--sparse-fill",
            action="store_true",
            default=False,
            help="Fill the sparse data with zeros",
        )
        _add_variant_arg(prs)

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        from unipercept.render import plot_input_data
        from unipercept.vision.point import sparse_fill

        ds = _get_dataset_by_hash(args)
        queue, pipe = ds()

        for i_arg in args.index:
            if i_arg == "?":
                i: int = random.randint(0, len(queue) - 1)
            else:
                i = int(i_arg)
            sample_id, queue_item = queue[i]

            table = create_table(
                {"id": sample_id, **queue_item},
                format="long",
            )
            print(table, file=sys.stdout, flush=True)
            _, sample = pipe[i]
            print(sample, file=sys.stdout, flush=True)

            seg = sample.captures.segmentations
            if seg is not None:
                print(f"Semantic classes : {seg.get_semantic_map().unique().tolist()}")
                print(f"Tracklet IDs     : {seg.get_instance_map().unique().tolist()}")

                if args.sparse_fill:
                    sample.captures.segmentations = sparse_fill(seg, seg >= 0)
            dep = sample.captures.depths
            if dep is not None:
                print(
                    f"Depth range      : {dep[dep > 0].min().item()} - {dep.max().item()}"
                )
                dep_nan = dep.where(dep > 0, torch.nan)
                dep_mean = dep_nan.nanmean()
                print(f"Depth mean       : {dep_mean.item()}")

                if args.sparse_fill:
                    sample.captures.depths = sparse_fill(dep, dep > 0)
            fig = plot_input_data(sample, info=ds.info)
            if not args.output:
                render.terminal.show(fig)
            else:
                fig.savefig(args.output)


class DownloadSubcommand(Subcommand, name="download"):
    """
    Download a dataset (if supported).
    """

    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument(
            "--force",
            "-f",
            action="store_true",
            help="force download even if the dataset is already downloaded",
        )
        _add_datasets_arg(prs)

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        from unipercept.data.sets import catalog

        for ds_key in args.datasets:
            ds_cls = catalog.get(ds_key)
            for variant in ds_cls.variants():
                print("Downloading %s(%s)...", ds_key, _kwargs_to_str(variant))
                ds_cls(**variant).download(force=args.force)


class ListSubcommand(Subcommand, name="list"):
    """
    List available datasets.
    """

    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        pass

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        from unipercept.data.sets import catalog

        for ds in catalog.list_datasets():
            print(ds)


class ListVariantsSubcommand(Subcommand, name="list-variants"):
    """
    List available dataset variants.
    """

    @staticmethod
    @TX.override
    def setup(prs: argparse.ArgumentParser):
        _add_datasets_arg(prs, nargs="*")

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        from unipercept.data.sets import catalog

        df_list: list[pd.DataFrame] = []

        if len(args.datasets) == 0:
            datasets = catalog.list_datasets()
        else:
            datasets = args.datasets

        for ds_key in datasets:
            ds_cls = catalog.get(ds_key)
            info_list = []
            for variant in ds_cls.variants():
                ds = ds_cls(**variant)

                ds_attrs = {}
                for k, v in variant.items():
                    ds_attrs[k] = v
                for field in D.fields(ds):
                    if field.name.startswith("_"):
                        continue
                    if field.name in ds_attrs:
                        continue
                    ds_attrs[field.name] = getattr(ds, field.name)

                info_list.append(
                    {
                        "id": ds_key,
                        "hash": ds.hash,
                        "class": generate_path(ds_cls),
                        **ds_attrs,
                    }
                )
            df = pd.DataFrame(data=info_list)
            df_list.append(df)

        df_all = pd.concat(df_list, ignore_index=True)
        df_all.drop(columns=["queue_fn"], inplace=True)
        df_all.fillna("", inplace=True)

        print(create_table(df_all, format="wide"))


class StatsSubcommand(Subcommand, name="stats"):
    """
    Show statistics about the number of items in a dataset.
    """

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
        from unipercept.data.sets import Manifest, catalog

        st_list: list[dict[str, T.Any]] = []
        for ds_key in args.dataset:
            ds_cls = catalog.get(ds_key)
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
    """
    Show information about a dataset, also known as 'metadata'.
    """

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
            "--purge",
            action="store_true",
            help="purge the cache (if it exists)",
        )
        prs.add_argument("dataset", help="dataset name")

    @staticmethod
    @TX.override
    def main(args: argparse.Namespace):
        """
        Cache the dataset.
        """
        from unipercept.data.sets import catalog

        ds_cls = catalog.get(args.dataset)

        stats = []
        for variant in ds_cls.variants():
            ds = ds_cls(**variant)
            path_str = ds.cache_path
            path = file_io.Path(path_str)

            exists = path.is_file()
            if args.purge and exists:
                path.unlink()
                cached = "True (purged)"
            else:
                cached = str(exists)

            stats.append(
                {
                    "dataset": args.dataset,
                    **variant,
                    "cached": cached,
                    "path": path_str,
                }
            )

        df = pd.DataFrame(stats)
        print(create_table(df, format="wide"))


if __name__ == "__main__":
    command.root("datasets")
