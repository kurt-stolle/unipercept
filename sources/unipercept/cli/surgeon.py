r"""
Weight surgeon CLI.
"""

from __future__ import annotations

import argparse
import pickle
import typing as T

import pandas as pd
import regex as re
import safetensors.torch
import torch

import unipercept as up
from unipercept.cli._command import command, logger
from unipercept.log import create_table

__all__ = []

Subcommand = up.utils.cli.create_subtemplate()


def _read_weights(arg: str, *, raw: bool = True) -> dict[str, torch.Tensor]:
    path = up.file_io.Path(arg)
    match path.suffix.lower():
        case ".safetensors":
            obj = safetensors.torch.load_file(path, device="cpu")
        case ".pth":
            obj = torch.load(path, map_location="cpu")
        case ".pkl":
            with up.file_io.Path.open(path, "rb") as fh:
                obj = pickle.load(fh)
        case _:
            msg = f"Unsupported file format: {path}"
            raise ValueError(msg)
    if not isinstance(obj, T.Mapping):
        raise ValueError(f"Expected a mapping, got {type(obj)}")
    if not raw:
        return obj
    metadata = {}
    for key in list(obj.keys()):
        if key.startswith("__"):
            metadata[key] = obj.pop(key)
    if len(metadata) > 0:
        logger.info("Found metadata: \n%s", create_table(metadata, format="auto"))
    if "state_dict" in obj:
        logger.info("Found 'state_dict' key. Possibly a PyTorch model checkpoint")
        obj = obj["state_dict"]
    if "model" in obj:
        logger.info("Found 'model' key. Possibly a Detectron2 model checkpoint")
        obj = obj["model"]
    if len(obj) == 1:
        key = next(iter(obj.keys()))
        logger.info("Found a single key: %s", key)
        obj = obj[key]
    return obj


def _import_weights(obj: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(obj, T.Mapping):
        raise ValueError(f"Expected a mapping, got {type(obj)}")
    if not all(isinstance(key, str) for key in obj):
        raise ValueError("All keys must be strings, got mixed types")
    for key, value in obj.items():
        if not isinstance(value, torch.Tensor):
            yield key, torch.as_tensor(value)
        else:
            yield key, value


def _add_weights_arg(prs: argparse.ArgumentParser):
    r"""Adds the `weights` argument to the parser."""

    prs.add_argument(
        "--match", "-m", type=re.compile, default=None, help="filters weights by name"
    )
    prs.add_argument(
        "weights",
        type=_read_weights,
        help="model state dict file, e.g. `model.pth` or `model.safetensors`",
    )


class InspectSubcommand(Subcommand, name="inspect"):
    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        _add_weights_arg(prs)
        prs.add_argument(
            "--depth",
            "-d",
            type=int,
            default=4,
            help="depth of the structure to display",
        )
        prs.add_argument(
            "--items",
            "-i",
            type=int,
            default=8,
            help="maximum amount of items to display per level when the object contains a sequence",
        )

    @staticmethod
    @T.override
    def main(args: argparse.Namespace):
        def cat_structure(
            m: T.Mapping[str, T.Any], depth: int = 2, current_depth: int = 0
        ) -> T.Generator[tuple[int, str, str], None, None]:
            for key, value in m.items():
                yield (
                    current_depth,
                    key,
                    type(value).__qualname__
                    if not isinstance(value, str)
                    else f"'{value}'",
                )
                if current_depth < depth:
                    if isinstance(value, T.Mapping):
                        # if _is_state_dict(value):
                        #    yield (current_depth, key, "<weights>")
                        yield from cat_structure(value, depth, current_depth + 1)
                    elif isinstance(value, T.Iterable) and not isinstance(
                        value, (str, bytes)
                    ):
                        shape = getattr(value, "shape", None)
                        dtype = getattr(value, "dtype", None)
                        if shape is not None or dtype is not None:
                            yield (current_depth, key, f"<{shape} {dtype}>")
                            continue
                        unique_types = set(type(item) for item in value)
                        if len(unique_types) == 1:
                            item_type = unique_types.pop()
                            item_repr = item_type.__qualname__
                            yield (
                                current_depth,
                                key,
                                f"<{item_repr} x {len(value)}>",
                            )
                        else:
                            yield (current_depth, key, "<mixed>")

        if not isinstance(args.weights, T.Mapping):
            logger.error("Input is not a mapping")
            return

        if args.depth > 0:
            logger.debug("Walking structure of input object...")
            struct = []
            cur_depth = 0
            cur_items = args.items

            for depth, key, value_type in cat_structure(args.weights):
                if "OrderedDict" in value_type:
                    cur_items = args.items
                    cur_depth = depth + 1
                elif depth != cur_depth:
                    cur_items = args.items
                    cur_depth = depth
                elif cur_items >= 0:
                    cur_items -= 1
                elif cur_items == 0:
                    struct.append(
                        "\t" * depth + "... (use --items/-i to display more)"
                    )
                    cur_items -= 1
                    continue
                else:
                    continue
                struct.append("\t" * depth + f"{key}: {value_type}")
            if len(struct) == 0:
                struct = ["<empty>"]
        else:
            struct = ["<skipped>"]
        logger.info(
            "Object structure: \n%s\nPass --depth/-d to control recursion depth (current: %d)\n\n",
            "\n".join(struct),
            args.depth,
        )

        logger.info("Number of keys : %d", len(list(args.weights.keys())))
        logger.info("Input type     : %s", type(args.weights))


class StatsSubcommand(Subcommand, name="stats"):
    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        _add_weights_arg(prs)
        prs.add_argument(
            "--negate",
            "-n",
            default=False,
            action="store_true",
            help="negates the filter",
        )

    @staticmethod
    @T.override
    def main(args: argparse.Namespace):
        data = []

        logger.info("Computing statistics for weights...")

        total_params = 0
        total_trainable_params = 0

        for name, weight in _import_weights(args.weights):
            if args.match:
                if args.match.search(name):
                    if args.negate:
                        continue
                elif args.negate:
                    pass
                else:
                    continue
            rec = {
                "weight": name,
                "shape": tuple(weight.shape),
                "dtype": str(weight.dtype).split(".")[-1],
                "min": weight.min().item(),
                "max": weight.max().item(),
                "mean": (
                    weight.mean().item() if torch.is_floating_point(weight) else None
                ),
                "std": weight.std().item() if torch.is_floating_point(weight) else None,
            }
            data.append(rec)

            total_params += weight.numel()
            if weight.requires_grad:
                total_trainable_params += weight.numel()

        data = pd.DataFrame(data)
        print(up.log.create_table(data, format="wide"))

        logger.info("Found %d weights", len(data))
        logger.info(
            "Total parameters: %d (%d trainable)", total_params, total_trainable_params
        )


class ExtractCommand(Subcommand, name="extract"):
    r"""
    Extracts weights from a model state dict.
    """

    @staticmethod
    @T.override
    def setup(prs: argparse.ArgumentParser):
        prs.add_argument("--output", "-o", type=up.file_io.Path, help="output file")
        prs.add_argument(
            "--insert",
            "-i",
            action="store_true",
            help="inserts the weights into the output file",
        )
        prs.add_argument(
            "--force",
            "-f",
            action="store_true",
            help="forces overwriting the output file or inserting the weights that already exist",
        )
        g = prs.add_mutually_exclusive_group()
        g.add_argument(
            "--negate",
            "-n",
            default=False,
            action="store_true",
            help="negates the filter",
        )
        g.add_argument(
            "--replace", "-r", type=str, default=None, help="replaces matched names"
        )
        _add_weights_arg(prs)

    @staticmethod
    @T.override
    def main(args: argparse.Namespace):
        if args.insert and not args.output:
            logger.info("Provide an output path with `--output` to insert the weights")
            return

        res = {}
        logger.info(
            "Selecting weights with match=%r",
            args.match.pattern if args.match else None,
        )
        for name, weight in _import_weights(args.weights):
            if args.match:
                match = args.match.search(name)
                if not match:
                    if args.negate:
                        res[name] = weight
                    continue
                if args.replace is not None:
                    assert (
                        args.negate is False
                    ), "Cannot negate and replace at the same time"
                    name = re.sub(args.match, args.replace, name).strip(" .")
                    if len(name) == 0:
                        logger.error("Empty name after replacement")
                        return
                if args.negate:
                    continue
                res[name] = weight

        if len(res) == 0:
            logger.error("No weights matched the filter")
            return
        for k in res:
            print(k)
        logger.info("Selected %d weights", len(res))

        if not args.output:
            logger.info(
                "Provide an output path with `--output` to save the altered weights"
            )
            return
        if args.output.exists():
            if not args.force and not args.insert:
                logger.error("File already exists: %s", args.output)
                return
            if args.insert:
                tgt = _read_weights(args.output)
                if not args.force:
                    tgt_keys = set(tgt.keys())
                    res_keys = set(res.keys())

                    if tgt_keys & res_keys:
                        logger.error("Weights already exist in the target file")
                        return
                tgt.update(res)
                res = tgt
                logger.info("Merged with %d existing weights", len(tgt))

                for k in res.keys():
                    print(k)
                logger.info("Total %d weights", len(res))
        if args.output.suffix.lower() == ".safetensors":
            safetensors.torch.save_file(res, args.output)
        elif args.output.suffix.lower() in (".pth", ".pt"):
            torch.save(res, args.output)
        else:
            msg = f"Unsupported file format: {args.output}"
            raise ValueError(msg)

        logger.info("Saved to %s", str(args.output))


command_name = up.file_io.Path(__file__).stem
command(command_name, help="model weight surgeon")(Subcommand)
if __name__ == "__main__":
    command.root(command_name)
