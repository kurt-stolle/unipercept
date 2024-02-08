"""
Commands that describe datasets.
"""

from __future__ import annotations

import argparse
import dataclasses as D
import sys
import typing as T

from tabulate import tabulate_formats

from unipercept import file_io, render
from unipercept.cli._command import command, logger
from unipercept.data.sets import PerceptionDataset

__all__ = []

# @command(help="describe a dataset to std")
# def datasets(parser):
#     parser.add_argument(
#         "--format", default="pprint", help="output format", choices=["yaml", "pprint"]
#     )
#     parser.add_argument(
#         "--output",
#         "-o",
#         help="directory to store output data and visualizations",
#         default=None,
#         type=file_io.Path,
#     )

#     mp = parser.add_mutually_exclusive_group()
#     mp.add_argument("--info", help="show dataset info", action="store_true")
#     mp.add_argument("--manifest", help="show dataset manifest", action="store_true")
#     mp.add_argument(
#         "--sequences", help="show information about the sequences", action="store_true"
#     )
#     mp.add_argument(
#         "--frames", help="show information about the frames", action="store_true"
#     )
#     mp.add_argument("--depth", help="show depth statistics", action="store_true")

#     parser.add_argument("dataset", help="dataset name")

#     return main


# def extract_depth_stats(ds: PerceptionDataset, output_dir: file_io.Path | None = None):
#     dep_min = float("inf")
#     dep_max = float("-inf")

#     loader = torch.utils.data.DataLoader(
#         ds.datapipe,
#         batch_size=1,
#         num_workers=0,
#         pin_memory=False,
#         prefetch_factor=4,
#         persistent_workers=False,
#     )

#     for inputs in loader:
#         if output_dir is not None:
#             img = render.plot_input_data(inputs, info=ds.info)
#             img.save(output_dir / f"{inputs.captures.primary_key}.png")

#         dep = inputs.captures.depths
#         dep = dep[dep > 0]

#         dep_min = dep.min().item()
#         dep_max = dep.max().item()

#         break

#     prog = tqdm(loader)
#     for inputs in prog:
#         dep = inputs.captures.depths
#         dep = dep[dep > 0]

#         dep_min = min(dep_min, dep.min().item())
#         dep_max = max(dep_max, dep.max().item())

#         prog.set_postfix({"min": dep_min, "max": dep_max})

#     print(f"Min: {dep_min}, Max: {dep_max}")


# def handle_request(args) -> T.Any:
#     if args.info:
#         from unipercept import get_info

#         return get_info(args.dataset)
#     else:
#         import inspect

#         import yaml

#         from unipercept import get_dataset

#         ds_cls = get_dataset(args.dataset)

#         sig = inspect.signature(ds_cls)
#         params = sig.parameters

#         # Inform user about available keyword arguments, with their default values
#         available_kwargs = {}
#         for name, param in params.items():
#             if param.kind == inspect.Parameter.KEYWORD_ONLY:
#                 available_kwargs[name] = (
#                     param.default if param.default != inspect.Parameter.empty else None
#                 )

#         logger.info(
#             f"Required arguments: %s",
#             ", ".join(
#                 [name for name, default in available_kwargs.items() if default is None]
#             ),
#         )

#         kwargs = input("Enter keyword arguments as YAML object:\n")
#         kwargs = yaml.safe_load(kwargs)

#         ds = ds_cls(**kwargs)

#         if args.depth:
#             return extract_depth_stats(ds, args.output)
#         elif args.manifest:
#             return ds.manifest
#         elif args.sequences:
#             mfst = ds.manifest

#             seqs = {}
#             for seq_id, seq in mfst["sequences"].items():
#                 seqs[seq_id] = "{} captures @ {} fps".format(
#                     len(seq.get("captures")), seq.get("fps", "???")
#                 )

#             return seqs
#         elif args.frames:
#             mfst = ds.manifest

#             frames = {}
#             for seq_id, seq in mfst["sequences"].items():
#                 for frame in seq["captures"]:
#                     frames[frame["primary_key"]] = "Ground truths: {}".format(
#                         ", ".join(frame["sources"].keys())
#                     )

#             return frames
#         else:
#             from dataclasses import asdict

#             return asdict(ds)


# def main(args):
#     try:
#         res_dict = handle_request(args)
#     except KeyError as e:
#         from unipercept.data.sets import list_datasets

#         logger.info(f"Unknown dataset: {e}")
#         avail = list_datasets()

#         if len(avail) == 0:
#             logger.info("No datasets available.")
#         else:
#             avail_str = "\n\t- ".join(list_datasets())
#             logger.info(f"Available datasets: \n\t- {avail_str}")
#         return

#     format = args.format
#     if format == "pprint":
#         from pprint import pformat
#         from shutil import get_terminal_size

#         res = pformat(
#             res_dict,
#             indent=1,
#             compact=False,
#             depth=2,
#             width=get_terminal_size().columns - 1,
#         )
#         logger.info("Result (Python): %s", res)
#     elif format == "yaml":
#         import yaml

#         res = yaml.dump(res_dict, allow_unicode=True, default_flow_style=False)
#         logger.info("Result (YAML): %s", res)
#     else:
#         print(f"Unknown format: {format}")


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
    subprs.add_parser("ls", help="list available datasets")

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
        case "ls":
            _main_ls()
        case "info":
            _main_info(args)
        case _:
            msg = f"Unknown subcommand: {args.datasets_subcommand}"
            raise ValueError(msg)


def _main_ls():
    """
    List available datasets.
    """
    from unipercept.data.sets import catalog

    for ds in catalog.list_datasets():
        print(ds)


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
