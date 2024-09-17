from __future__ import annotations

import sys

from unipercept.cli._command import command
from unipercept.config import lazy

__all__ = []


def main(args):
    from omegaconf import OmegaConf

    args_dict = vars(args)
    args_dict.pop("func")

    fmt = args_dict.pop("format")
    out = args_dict.get(args_dict.pop("key"))
    if fmt == "json":
        import json

        args_dict["config"] = OmegaConf.to_object(args.config)
        res = json.dumps(out, indent=4, ensure_ascii=False)
        print(res, file=sys.stdout, flush=True)
    elif fmt == "yaml":
        config_yaml = lazy.dump_config(args.config)
        print(config_yaml, file=sys.stdout, flush=True)
    elif fmt == "python":
        config_yaml = lazy.yaml
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)


@command(help="output the configuration and arguments list to stdout")
@command.with_config
def echo(parser):
    parser.add_argument(
        "--format",
        default="yaml",
        help="output format",
        choices=["yaml", "python", "json"],
    )
    parser.add_argument("--key", default="config", help="key to output")

    return main


if __name__ == "__main__":
    command.root("echo")
