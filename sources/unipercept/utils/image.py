"""
It is used to determine the file format of the images in the dataset.

This file is an extended version Python 3.11 standard library,
this was done because the module will be removed in Python 3.13.

Source: https://raw.githubusercontent.com/python/cpython/3.11/Lib/imghdr.py

Added featues are:
- enumerations for image formats.
- a function to read the image size from the file header.
"""

import struct
from enum import StrEnum
from enum import auto as A
from os import PathLike
from pathlib import Path
from typing import IO, NamedTuple, Optional

__all__ = ["what", "Format", "size", "Size"]


# ------------------------------------ #
# Determines the type of an image file #
# ------------------------------------ #


class Format(StrEnum):
    JPEG = A()
    PNG = A()
    GIF = A()
    TIFF = A()
    BMP = A()
    PBM = A()
    PGM = A()
    PPM = A()
    RAST = A()
    XBM = A()
    WEBP = A()
    EXR = A()
    RGB = A()


def what(file: str | PathLike | IO, h=None) -> Optional[Format]:
    """
    Tries to determine the type of an image file given its file path
    """
    f = None
    try:
        if h is None:
            if isinstance(file, (str, PathLike)):
                f = open(file, "rb")
                h = f.read(32)
            else:
                location = file.tell()
                h = file.read(32)
                file.seek(location)
        for tf in tests:
            res = tf(h, f)
            if res:
                return res
    finally:
        if f:
            f.close()
    return None


# ------------------------------- #
# Subroutines per image file type #
# ------------------------------- #

tests = []


def test_jpeg(h, f):
    """JPEG data with JFIF or Exif markers; and raw JPEG"""
    if h[6:10] in (b"JFIF", b"Exif"):
        return Format.JPEG
    elif h[:4] == b"\xff\xd8\xff\xdb":
        return Format.JPEG


tests.append(test_jpeg)


def test_png(h, f):
    if h.startswith(b"\211PNG\r\n\032\n"):
        return Format.PNG


tests.append(test_png)


def test_gif(h, f):
    """GIF ('87 and '89 variants)"""
    if h[:6] in (b"GIF87a", b"GIF89a"):
        return Format.GIF


tests.append(test_gif)


def test_tiff(h, f):
    """TIFF (can be in Motorola or Intel byte order)"""
    if h[:2] in (b"MM", b"II"):
        return Format.TIFF


tests.append(test_tiff)


def test_rgb(h, f):
    """SGI image library"""
    if h.startswith(b"\001\332"):
        return Format.RGB


tests.append(test_rgb)


def test_pbm(h, f):
    """PBM (portable bitmap)"""
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"14" and h[2] in b" \t\n\r":
        return Format.PBM


tests.append(test_pbm)


def test_pgm(h, f):
    """PGM (portable graymap)"""
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"25" and h[2] in b" \t\n\r":
        return Format.PGM


tests.append(test_pgm)


def test_ppm(h, f):
    """PPM (portable pixmap)"""
    if len(h) >= 3 and h[0] == ord(b"P") and h[1] in b"36" and h[2] in b" \t\n\r":
        return Format.PPM


tests.append(test_ppm)


def test_rast(h, f):
    """Sun raster file"""
    if h.startswith(b"\x59\xA6\x6A\x95"):
        return Format.RAST


tests.append(test_rast)


def test_xbm(h, f):
    """X bitmap (X10 or X11)"""
    if h.startswith(b"#define "):
        return Format.XBM


tests.append(test_xbm)


def test_bmp(h, f):
    if h.startswith(b"BM"):
        return Format.BMP


tests.append(test_bmp)


def test_webp(h, f):
    if h.startswith(b"RIFF") and h[8:12] == b"webp":
        return Format.WEBP


tests.append(test_webp)


def test_exr(h, f):
    if h.startswith(b"\x76\x2f\x31\x01"):
        return Format.EXR


tests.append(test_exr)


# -------------------------------- #
# Determine image size from header #
# -------------------------------- #


class Size(NamedTuple):
    """
    Image size, in pixels (width, height)
    """

    width: int
    height: int


def size(path: Path | str) -> Size:
    """
    Read image size from the file headers; does not require file to be opened.
    This is useful in sitations where a fast assessment of image sizes must be
    made, like when providing metadata to a dataset record.

    Parameters
    ----------
    path
        Path to the image file.

    Returns
    -------
        Width and height of the image.
    """

    path = str(path)

    with open(path, "rb") as fh:
        head = fh.read(24)
        if len(head) != 24:
            raise ValueError(f"bad length read {len(head)} for {path}!")

        image_type = what(path)
        if image_type == "png":
            check = struct.unpack(">i", head[4:8])[0]
            if check != 0x0D0A1A0A:
                raise ValueError(f"Check failed: {check}!")
            width, height = struct.unpack(">ii", head[16:24])
        elif image_type == "gif":
            width, height = struct.unpack("<HH", head[6:10])
        elif image_type == "jpeg":
            try:
                fh.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xC0 <= ftype <= 0xCF:
                    fh.seek(size, 1)
                    byte = fh.read(1)
                    while ord(byte) == 0xFF:
                        byte = fh.read(1)
                    ftype = ord(byte)
                    size = struct.unpack(">H", fh.read(2))[0] - 2
                # We are at a SOFn block
                fh.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack(">HH", fh.read(4))
            except Exception:  # IGNORE:W0703
                raise
        else:
            raise NotImplementedError(image_type)

    return Size(width=width, height=height)


# --------------------#
# Small test program #
# --------------------#


def test():
    import sys

    recursive = 0
    if sys.argv[1:] and sys.argv[1] == "-r":
        del sys.argv[1:2]
        recursive = 1
    try:
        if sys.argv[1:]:
            testall(sys.argv[1:], recursive, 1)
        else:
            testall(["."], recursive, 1)
    except KeyboardInterrupt:
        sys.stderr.write("\n[Interrupted]\n")
        sys.exit(1)


def testall(list, recursive, toplevel):
    import os
    import sys

    for filename in list:
        if os.path.isdir(filename):
            print(filename + "/:", end=" ")
            if recursive or toplevel:
                print("recursing down:")
                import glob

                names = glob.glob(os.path.join(glob.escape(filename), "*"))
                testall(names, recursive, 0)
            else:
                print("*** directory (use -r) ***")
        else:
            print(filename + ":", end=" ")
            sys.stdout.flush()
            try:
                print(what(filename))
            except OSError:
                print("*** not found ***")


if __name__ == "__main__":
    test()
