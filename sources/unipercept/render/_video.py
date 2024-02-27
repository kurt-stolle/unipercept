from __future__ import annotations

import dataclasses as D
import functools
import os
import sys
import tempfile
import typing as T
from contextlib import contextmanager

import PIL.Image as pil_image

from unipercept.utils.typings import Pathable

__all__ = ["video_writer"]


@contextmanager
def video_writer(out: Pathable, *, fps: int, overwrite: bool = False):
    """
    Used for writing a sequence of PIL images to a (temporary) directory, and then
    encoding them into a video file using ``ffmpeg`` commands.
    """

    from unipercept.file_io import Path

    def _parse_output_path(out: Pathable) -> str:
        out = Path(out)
        if out.is_file():
            if not overwrite:
                msg = f"File {out!r} already exists, and overwrite is set to False."
                raise FileExistsError(msg)
            out.unlink()
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
        return str(out)

    def _get_ffmpeg_path() -> str:
        return "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    def _get_ffmpeg_cmd(fps: int, dir: str, out: str) -> tuple[str, ...]:
        frame_glob = os.path.join(dir, "*.png")
        return (
            _get_ffmpeg_path(),
            f"-framerate {fps}",
            "-pattern_type glob",
            f"-i {frame_glob!r}",
            "-c:v libx264",
            "-pix_fmt yuv420p",
            f"{out!r}",
        )

    def _save_image(im: pil_image.Image, *, dir: str):
        next_frame = len(os.listdir(dir))
        im.save(os.path.join(dir, f"{next_frame:010d}.png"))

    with tempfile.TemporaryDirectory() as dir:
        try:
            yield functools.partial(_save_image, dir=dir)
        finally:
            cmd = " ".join(_get_ffmpeg_cmd(fps, dir, out=_parse_output_path(out)))

            print(cmd, file=sys.stderr)
            os.system(cmd)
