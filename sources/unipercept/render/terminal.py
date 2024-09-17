"""
Utilites showing images over the terminal.

The following methods are supported:

    1. LibSixel (via ``img2sixel`` or the Python bindings)
    2. KiTTy (via ``icat``)
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import tempfile
import typing as T
import warnings

import matplotlib.pyplot as plt
import PIL.Image as pil_image

if T.TYPE_CHECKING:
    from unipercept.types import Pathable

_DISPLAY_HANDLERS: T.MutableMapping[str, T.Callable[[str], None]] = {}


def show(image: pil_image.Image | plt.Figure | Pathable) -> None:
    """
    Show an image over the terminal.

    Parameters
    ----------
    image : PIL.Image
        The image to show.
    """
    from unipercept.config.env import get_env
    from unipercept.file_io import Path

    if isinstance(image, plt.Figure):
        image.canvas.draw()
        image = pil_image.frombytes(
            "RGB", image.canvas.get_width_height(), image.canvas.tostring_rgb()
        )
    if isinstance(image, pil_image.Image):
        with tempfile.NamedTemporaryFile() as f:
            image.save(f, format="png")
            return show(f.name)

    if not isinstance(image, str):
        image = str(Path(image))

    handler_key = get_env(
        str, "UP_RENDER_TERMINAL_HANDLER", default=next(iter(_DISPLAY_HANDLERS.keys()))
    )
    if handler_key not in _DISPLAY_HANDLERS:
        msg = f"Terminal renderer {handler_key!r} not found. Choose from: {list(_DISPLAY_HANDLERS.keys())}"
        warnings.warn(msg, stacklevel=2)
    else:
        try:
            _DISPLAY_HANDLERS[handler_key](image)
        except Exception as e:
            warnings.warn(
                f"Failed to render image with {handler_key}: {e}", stacklevel=2
            )


def register_handler(
    name: str, *, test: T.Callable[[], bool] | None = None
) -> T.Callable[[T.Callable[[str], None]], None]:
    def register(fn):
        if test is None:
            test_passed = True
        else:
            try:
                test_passed = bool(test())
            except Exception:
                test_passed = False
        if test_passed:
            _DISPLAY_HANDLERS[name] = fn

        return fn

    return register


@register_handler("libsixel", test=lambda: importlib.import_module("libsixel.encoder"))
def _show_libsixel(path: str) -> None:
    """
    Show an image over the terminal using Sixel with the python bindings from
    `LibSixel <https://github.com/saioha/libsixel>`_.

    See Also
    --------
    - `Are We Sixel yet? <https://www.arewesixelyet.com>`_
    """
    from libsixel import encoder

    enc = encoder.Encoder()
    enc.setopt(encoder.SIXEL_OPTFLAG_HIGH_COLOR, 1)
    enc.encode(path)


@register_handler("img2sixel", test=lambda: shutil.which("img2sixel"))
def _show_img2sixel(path: str) -> None:
    """
    Show an image over the terminal using Sixel with the `img2sixel` command from
    `LibSixel <https://github.com/saioha/libsixel>`_.

    See Also
    --------
    - `Are We Sixel yet? <https://www.arewesixelyet.com>`_
    """
    subprocess.run(["img2sixel", "-I", path], check=False)


@register_handler("kitty", test=lambda: shutil.which("kitty"))
def _show_kitty(path: str) -> None:
    """
    Show an image over the terminal using KiTTY iCat.

    See Also
    --------
    - `kitty icat <https://sw.kovidgoyal.net/kitty/kittens/icat>`_
    """
    subprocess.run(["kitty", "+kitten", "icat", path], check=False)


@register_handler("x11")
def _show_x11(path: str) -> None:
    raise NotImplementedError()
