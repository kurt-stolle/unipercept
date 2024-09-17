"""
Run a model in realtime or on a previously saved directory of images.
"""

from __future__ import annotations

import argparse

import torch

import unipercept as up
from unipercept.cli._command import command

_logger = up.log.get_logger()


KEY_SESSION_ID = "session_id"


@command(help="run a model on a directory/stream of images", description=__doc__)
@command.with_config
def run(p: argparse.ArgumentParser):
    p_size = p.add_mutually_exclusive_group(required=False)
    p_size.add_argument(
        "--size", type=int, help="Size of the input images in pixels (smallest side)"
    )
    p.add_argument(
        "--weights",
        "-w",
        type=str,
        help="path to load model weights from (overrides any state recovered by the config)",
    )
    p.add_argument(
        "--render",
        type=str,
        default="segmentation",
        choices=["segmentation", "depth", "noop"],
        help="rendering mode",
    )

    p.add_argument("input", type=str, default="0", help="input stream or directory")

    return _main


@torch.inference_mode()
def _main(args: argparse.Namespace):
    model = up.create_model(args.config)
    preprocess = _build_transforms(args)

    if up.file_io.isdir(args.input):
        run = _run_filesystem(model, preprocess, args.input)
    else:
        cap_num = int(args.input)
        cap = _get_capture(cap_num)
        run = _run_realtime(model, preprocess, cap)

    for inp, out in run:
        print(out)


def _build_transforms(args):
    import torchvision.transforms.v2 as transforms

    tf = []
    if args.size:
        tf.append(up.data.ops.TorchvisionOp(transforms.Resize(args.size)))

    tf.append(up.data.ops.PadToDivisible(32))

    return tf


def _get_capture(cap_num):
    import cv2

    cap = cv2.VideoCapture(cap_num, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
    cap.set(cv2.CAP_PROP_FPS, 1)
    return cap


def _run_realtime(model, preprocess, cap):
    import torchvision.transforms.v2 as transforms

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    frame_num = 0
    while True:
        ret, img_np = cap.read()
        if not ret:
            break

        # BGR -> RGB
        img = transforms.functional.to_tensor(img_np[..., [2, 1, 0]])
        inp = up.create_inputs(img, frame_offset=frame_num)
        inp = preprocess(inp)
        out = model(inp)

        yield inp, out

        frame_num += 1


def _run_filesystem(model, preprocess, path):
    root = up.file_io.Path(path)
    root_paths = list(root.iterdir())

    if all(p.is_dir() for p in root_paths):
        for p in root_paths:
            yield from _run_filesystem(model, preprocess, p)
    elif all(p.name.endswith(".png") for p in root_paths):
        for p in root_paths:
            img = up.data.tensors.Image.read(p)
            inp = up.create_inputs(img, frame_offset=0)
            inp = preprocess(inp)
            out = model(inp)

            yield inp, out
    else:
        msg = (
            f"Invalid directory structure: {str(path)!r}, expected directories (sequence) "
            "of PNG images sortable by name!"
        )
        raise ValueError(msg)
