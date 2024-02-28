from __future__ import annotations

import dataclasses as D
import functools
import os
import subprocess
import sys
import tempfile
import typing
import typing as T
from contextlib import contextmanager

import PIL.Image as pil_image

from unipercept.log import get_logger
from unipercept.utils.typings import Pathable

__all__ = ["video_writer"]

_IgnoreExceptionsType: T.TypeAlias = T.Type[Exception] | T.Callable[[Exception], bool]

_logger = get_logger(__name__)


@contextmanager
def video_writer(
    out: Pathable,
    *,
    fps: int,
    overwrite: bool = False,
    ignore_exceptions: bool | _IgnoreExceptionsType = False,
):
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

    def _should_ignore(e: Exception) -> bool:
        if isinstance(ignore_exceptions, bool):
            return ignore_exceptions
        if isinstance(ignore_exceptions, type):
            return isinstance(e, ignore_exceptions)
        if callable(ignore_exceptions):
            return ignore_exceptions(e)
        msg = (
            "ignore_exceptions must be a bool, a (collection of) exception type(s), or "
            "a callable that takes an exception and returns a bool. "
            f"Got {ignore_exceptions!r} instead."
        )
        raise TypeError(msg)

    write_video = isinstance(ignore_exceptions, bool) and ignore_exceptions or False

    with tempfile.TemporaryDirectory() as dir:
        try:
            yield functools.partial(_save_image, dir=dir)
            write_video = True
        except Exception as e:  # noqa: PIE786
            if not _should_ignore(e):
                write_video = False
            if not write_video:
                raise
            _logger.warning("Ignoring exception: %s", e)
        finally:
            if write_video:
                cmd = " ".join(_get_ffmpeg_cmd(fps, dir, out=_parse_output_path(out)))
                _logger.debug("Writing video: %s", cmd)
                res = subprocess.run(cmd, shell=True, capture_output=True)
                if res.returncode != 0:
                    msg = f"Failed to write video: {res}"
                    _logger.error(msg)
            else:
                _logger.debug("Video writing was skipped.")
