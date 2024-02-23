from __future__ import annotations
import sys
from unipercept.cli._command import command

__all__ = []


def main(args):
    from omegaconf import OmegaConf

    args_dict = vars(args)
    args_dict["config"] = OmegaConf.to_object(args.config)
    args_dict.pop("func")

    fmt = args_dict.pop("format")
    out = args_dict.get(args_dict.pop("key"))
    if fmt == "pprint":
        from pprint import pformat
        from shutil import get_terminal_size

        if isinstance(out, dict):
            for key, value in out.items():
                head = f"-- {key} "
                print(
                    "\n" + head + "-" * (get_terminal_size().columns - len(head)),
                    end="\n\n",
                )
                res = pformat(
                    value,
                    indent=1,
                    compact=False,
                    depth=2,
                    width=get_terminal_size().columns - 1,
                )
                print(res)
        else:
            res = pformat(
                out,
                indent=1,
                compact=False,
                depth=2,
                width=get_terminal_size().columns - 1,
            )
            print(res)

    elif fmt == "json":
        import json 

        res = json.dumps(out, indent=4, ensure_ascii=False)
        print(res, file=sys.stdout, flush=True)

    elif fmt == "yaml":
        import yaml

        res = yaml.dump(out, allow_unicode=True, default_flow_style=False)
        print(res, file=sys.stdout, flush=True)
    else:
        print(f"Unknown format: {fmt}", file=sys.stderr)


@command(help="output the configuration and arguments list to stdout")
@command.with_config
def echo(parser):
    parser.add_argument(
        "--format", default="pprint", help="output format", choices=["yaml", "pprint", "json"]
    )
    parser.add_argument("--key", default="config", help="key to output")

    return main


if __name__ == "__main__":
    command.root("echo")
