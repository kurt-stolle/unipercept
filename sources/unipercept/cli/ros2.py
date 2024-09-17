"""
Run UniPercept in training/inference as a ROS2 node. Not for production use.
"""

from __future__ import annotations

import argparse

try:
    import rclpy
except ImportError:
    pass  # TODO

import unipercept as up

from ._command import command


@command(help="run as a ros2 node", description=__doc__)
@command.with_config
def ros2(p: argparse.ArgumentParser):
    p.add_argument(
        "--size", type=int, help="Size of the input images in pixels (smallest side)"
    )
    p.add_argument(
        "--weights",
        "-w",
        type=str,
        help="path to load model weights from (overrides any state recovered by the config)",
    )

    return main


def main(args):
    model = up.create_model(args.config, weights=args.weights)


if __name__ == "__main__":
    command.root("ros2")
