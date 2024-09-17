from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path

import anytree
import pandas as pd


def _get_argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)

    return parser


def _is_root(parent: str) -> bool:
    """
    Returns `TRUE` if the parent label indicates that the current label is a root, i.e. when it is blank or `NA`.
    """
    return (
        parent is None or parent is pd.NA or not parent or str(parent).lower() == "nan"
    )


def _find_roots(labels: dict[str, str]) -> tuple[set[str], dict[str, str]]:
    roots = set()
    labels_out = copy.copy(labels)
    for label, parent in labels.items():
        label = str(label)
        parent = str(parent)
        print(label, parent)
        if _is_root(parent):
            labels_out.pop(label)
            roots.add(label)

    return roots, labels_out


def _build_tree(labels: dict[str, str]) -> Iterable[anytree.Node]:
    """
    Construct a tree of labels from a dataframe of labels and a dataframe of assignments.
    """

    nodes = {}
    roots, labels = _find_roots(labels)

    for label, parent in labels.items():
        if label not in nodes:
            nodes[label] = anytree.Node(label)
        if parent not in nodes:
            nodes[parent] = anytree.Node(parent)
        nodes[label].parent = nodes[parent]

    for root in roots:
        yield nodes[root]


def _read_labels(path: Path) -> Iterable[tuple[str, str]]:
    labels = pd.read_csv(path, delimiter=";")

    for label, parent, *_ in labels.itertuples(index=False):
        # Filter out bad labels (e.g. empty rows)
        if label is None or str(label).lower() in {"", "nan", "none"}:
            continue

        # Convert to string
        label = str(label)
        parent = str(parent)

        # Return pair of [label, parent]
        yield label, parent


if __name__ == "__main__":
    args = _get_argparser().parse_args()

    # Mapping label -> parent
    labels = dict(_read_labels(args.labels))

    # Convert to a tree
    label_roots = _build_tree(labels)
    for node in label_roots:
        print(anytree.RenderTree(node))
