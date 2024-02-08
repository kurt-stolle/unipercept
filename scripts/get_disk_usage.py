#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from tabulate import tabulate
from tqdm import tqdm

import unipercept as up

PATHS = ["//output/", "//cache/", "//scratch/", "//datasets/", "//configs/"]
LOGGER = up.log.get_logger(__name__)


def humanbytes(B: int | float) -> str:
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

    if B < KB:
        return "{0} {1}".format(B, "Bytes" if 0 == B > 1 else "Byte")
    elif KB <= B < MB:
        return "{0:.2f} KB".format(B / KB)
    elif MB <= B < GB:
        return "{0:.2f} MB".format(B / MB)
    elif GB <= B < TB:
        return "{0:.2f} GB".format(B / GB)
    elif TB <= B:
        return "{0:.2f} TB".format(B / TB)

    return "?"


def get_size(path_str: str) -> tuple[up.file_io.Path, int]:
    path = up.file_io.Path(path_str)
    total = 0
    prog = tqdm(
        (f for f in path.glob("**/*") if f.is_file()),
        unit="files",
        postfix="0 B",
        desc=f"{path_str:16s}",
    )
    for f in prog:
        size = f.stat().st_size
        total += size

        prog.set_postfix_str(humanbytes(total))
    prog.close()
    return path, humanbytes(total)


if __name__ == "__main__":
    table = []

    for path_str in PATHS:
        path = up.file_io.Path(path_str).resolve()
        table.append((path_str, *get_size(path_str)))

    print(tabulate(table, tablefmt="simple", headers=["Canonical", "Expanded", "Size"]))
