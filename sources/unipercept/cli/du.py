"""
List the disk usage (akin to the `du -d 0 <path>` command) of a path.

Supports the prefixes registed in `unipercept.file_io`. By default, all prefixes
registered through the environment are listed.
"""

from __future__ import annotations

from sys import stdout

from tabulate import tabulate
from tqdm import tqdm

from unipercept import file_io

from ._command import command

DEFAULT_PREFIXES = ["//output/", "//cache/", "//scratch/", "//datasets/"]


@command("du")
def du(p: argparse.ArgumentParser):
    p.add_argument(
        "--format",
        "-F",
        type=str,
        default="simple",
        help="Control the format of the table printed to stdout",
    )
    p.add_argument(
        "--bytes",
        "-b",
        action="store_true",
        default=False,
        help="Show the amount of bytes as an integer.",
    )
    p.add_argument(
        "paths",
        nargs="*",
        type=str,
        default=DEFAULT_PREFIXES,
        help="Paths from which to determine disk usage.",
    )

    def main(args):
        table = ((path, *_discover_stats(path, not args.bytes)) for path in args.paths)
        columns = ("path", "location", "files", "directories", "ionodes", "size")
        result = tabulate(table, tablefmt=args.format, headers=columns)
        print(result, file=stdout, flush=True)

    return main


def _to_readable(B: int | float) -> str:
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

    if B < KB:
        return f"{int(B)}"
    if KB <= B < MB:
        return f"{B / KB:.2f} K"
    if MB <= B < GB:
        return f"{B / MB:.2f} M"
    if GB <= B < TB:
        return f"{B / GB:.2f} G"
    if TB <= B:
        return f"{B / TB:.2f} T"
    raise RuntimeError()


def _discover_stats(
    path_str: str, human_readable: bool
) -> T.Tuple[str, str, str, str, str]:
    path = file_io.Path(path_str)
    total_files: int = 0
    total_dirs: int = 0
    total_size: int = 0
    with tqdm(
        unit=" files",
        postfix="0 B",
        desc=f"{path_str:16s}",
    ) as prog:
        for f in path.glob("**/*"):
            if f.is_dir():
                total_dirs += 1
            else:
                total_size += f.stat().st_size
                total_files += 1

                prog.update(1)
                prog.set_postfix_str(_to_readable(total_size) + "B")
    total_nodes = total_files + total_dirs

    out_path = str(path)
    if human_readable:
        return (
            out_path,
            *map(_to_readable, (total_files, total_dirs, total_nodes)),
            (_to_readable(total_size) + "B"),
        )

    return out_path, *map(str, (total_files, total_dirs, total_nodes, total_size))
