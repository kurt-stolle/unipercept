from __future__ import annotations

from unipercept.cli._command import command

__all__ = []


@command(help="list available backbones")
def backbones(parser):
    parser.add_argument(
        "--pretrained", "-p", action="store_true", help="list only pretrained backbones"
    )
    parser.add_argument(
        "framework",
        default=["torchvision", "timm"],
        nargs="*",
        help="framework to list backbones of, choices: [torchvision, timm]",
    )
    return main


def main(args):
    import tabulate

    records = []

    for name in args.framework:
        for backbone in read_backbones(name, args.pretrained):
            records.append([name, backbone])

    print(tabulate.tabulate(records, headers=["framework", "backbone"]))


def read_backbones(name: str, pretrained: bool):
    import unipercept as up

    match name:
        case "torchvision":
            yield from up.nn.backbones.torchvision.list_available(pretrained=pretrained)
        case "timm":
            yield from up.nn.backbones.timm.list_available(pretrained=pretrained)
        case _:
            raise ValueError(f"Unknown framework '{name}'")


if __name__ == "__main__":
    command.root("backbones")
