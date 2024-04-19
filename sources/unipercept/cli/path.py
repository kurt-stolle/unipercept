"""
Simple interface to ``unipercept.file_io`` to get the path of a file or directroy.
"""

from __future__ import annotations

import argparse

from ._command import command


@command("path")
def path(prs: argparse.ArgumentParser):
    """
    Get the path of a file or directory.
    """
    prs.add_argument("path", type=str, nargs="+", help="path to the file or directory")

    def main(args):
        from unipercept import file_io

        for p in args.path:
            print(str(file_io.Path(p)))

    return main


if __name__ == "__main__":
    command.root("path")
