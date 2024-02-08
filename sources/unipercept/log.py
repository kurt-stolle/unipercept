"""
Canonical logger used throughout the project.
"""

from __future__ import annotations

import atexit
import functools
import io
import logging
import os
import sys
import time
import typing as T
from collections import Counter
from typing import Optional

from tabulate import tabulate
from termcolor import colored
from typing_extensions import override

__all__ = []

# Logging levels are defined by integer values, this mapping makes it easier to use strings, e.g. in a CLI setting.
LOG_LEVELS: T.Final[dict[str, int]] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "passive": -1,
}

# Default logging level, can be overridden by setting the UNI_LOG_LEVEL environment variable.
DEFAULT_LEVEL: T.Final[str] = os.getenv("UNI_LOG_LEVEL", "debug")


def log_every_n(
    level: str, message: str, n: int = 1, *, name: str | None = None
) -> None:
    """
    Log once per n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    from unipercept.utils.inspect import caller_identity, calling_module_name

    caller_module, key = caller_identity()
    _log_counter[key] += 1
    if n == 1 or _log_counter[key] % n == 1:
        logging.getLogger(name or caller_module).log(_get_level(level), message)


def log_every_n_seconds(
    level: str | int, message: str, n: int = 1, *, name: str | None = None
):
    """
    Log no more than once per n seconds.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = caller_identity()
    last_logged = _log_timers.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(_get_level(level), message)
        _log_timers[key] = current_time


def create_table(
    mapping: T.Mapping[T.Any, T.Any], format: T.Literal["long", "wide"] = "wide"
) -> str:
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Parameters
    ----------
    mapping
        Items to tabulate

    Returns
    -------
    The table as a string.
    """
    keys, values = tuple(zip(*mapping.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def get_logger(name: Optional[str] = None, **kwargs: T.Any) -> logging.Logger:
    """
    Get a logger instance, where the name is automatically set to the calling module name.
    """
    from unipercept.utils.inspect import caller_identity, calling_module_name

    return _get_handler(
        _canonicalize_name(name or calling_module_name(left=1)),
        **kwargs,
    )


def _canonicalize_name(name: str) -> str:
    """
    Canonicalizes a logger name into a full module name without private submodules.
    """
    from pathlib import Path

    if name.endswith(".py"):
        file = Path(name)
        root = Path(__file__).parent
        if file.is_relative_to(root):
            name = file.relative_to(root).with_suffix("").as_posix()
            name = name.replace("/", ".")
        else:
            raise ValueError(f"Cannot canonicalize name: {name}")

    try:
        name_parts = name.split(".")
        while name_parts[-1].startswith("_"):
            name_parts.pop()
        name = ".".join(name_parts)
    except IndexError:
        name = "unipercept"
    return name


def _get_level(name_or_code: int | str) -> int:
    """
    Convert a log level name or code to a code.
    """
    match name_or_code:
        case int():
            return name_or_code
        case str():
            return LOG_LEVELS[name_or_code.lower()]
        case _:
            raise TypeError(f"Unknown log level: {name_or_code} ({type(name_or_code)})")


@functools.lru_cache(maxsize=None)
def _get_handler(
    name: str,
    /,
    *,
    output=None,
    color=True,
    propagate: bool = False,
    stdout: bool = True,
    level: str | int = DEFAULT_LEVEL,
):
    """
    Get a logger with a default verbose formatter, cached to ensure that the same logger handler is shared across
    all the sites that call this function.
    """
    from unipercept import file_io
    from unipercept.state import get_process_index

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate

    # Determine the full name and the short name
    root_name, *sub_name = name.split(".", 1)

    # Translate the logging level
    level = _get_level(level)

    # Find the process index in distributed settings
    process_rank = max(0, get_process_index(local=False))

    # Logging to stdout
    if stdout and process_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(_get_formatter(root_name, color))

        logger.addHandler(ch)

    # Logging to file
    if output is not None:
        filename = file_io.Path(output)
        if filename.suffix in (".txt", ".log"):
            filename = filename
        elif filename.suffix == "":
            filename = filename / "log.txt"
        else:
            raise ValueError(f"Unknown log file extension: {filename.suffix}")
        filename.with_suffix(f".{process_rank}.{filename.suffix}")
        filename.parent.mkdir(exist_ok=True, parents=True)

        fh = logging.StreamHandler(_get_stream(filename))
        fh.setLevel(level)
        fh.setFormatter(_get_formatter(full_name, short_name, False))

        logger.addHandler(fh)

    return logger


@functools.lru_cache(maxsize=None)
def _get_formatter(root_name: str, color: bool = True) -> logging.Formatter:
    """
    Get a formatter instance, cached to ensure that the same formatter is shared across all the sites that call this.
    """

    datefmt = r"%Y-%m-%d %H:%M:%S"
    if color:
        prefix = "%(asctime)s %(name)s" + colored(":", attrs=("bold",))
        formatter = _ColorfulFormatter(
            prefix + " " + "%(message)s",
            datefmt=datefmt,
            root_name=root_name,
        )
    else:
        formatter = logging.Formatter(
            "[ %(asctime)s @ %(name)s ] (%(levelname)s) : %(message)s", datefmt=datefmt
        )

    return formatter


@functools.lru_cache(maxsize=None)
def _get_stream(filename) -> io.IOBase:
    """
    Taken from the `detectron` implementation.
    """

    from unipercept import file_io

    # use 1K buffer if writing to cloud storage
    io = file_io.open(filename, "a", buffering=1024**2)
    atexit.register(io.close)
    return io


class _ColorfulFormatter(logging.Formatter):
    """
    A logging formatter that supports colored output.
    Based on `detectron2.utils.logger._ColorfulFormatter`.
    """

    def __init__(self, *args, root_name: str, **kwargs):
        self._root_name = root_name + "."

        super().__init__(*args, **kwargs)

    @override
    def formatMessage(self, record):
        date, time = record.asctime.split(" ")
        record.asctime = (
            colored(date, color="cyan") + " " + colored(time, color="light_cyan")
        )

        match record.levelno:
            case logging.DEBUG:
                level_icon = "🐛"
            case logging.WARNING:
                level_icon = colored("⚠️", attrs=["blink"])
            case logging.ERROR:
                level_icon = colored("❌", attrs=["blink"])
            case logging.CRITICAL:
                level_icon = colored("😵", attrs=["blink"])
            case _:
                level_icon = "📝"

        root, *sub = record.name.split(".", 1)
        package_color = "yellow" if root == "unipercept" else "light_yellow"
        record.name = colored(root, attrs=["bold"], color=package_color)
        if sub and len(sub) > 0:
            record.name += colored("." + sub[0], color=package_color)
        record.name = f"{level_icon} {record.name}"  # {record.name:40s}"

        leader, sep, message = record.getMessage().partition(":")
        record.message = leader + sep + colored(message, attrs=["bold"])

        return super().formatMessage(record)


_log_counter = Counter()
_log_timers = {}


def log_first_n(
    level,
    message,
    n=1,
    *,
    name: str | None = None,
    key: tuple[str, ...] | str | T.Literal["caller", "message"] = "caller",
) -> None:
    """
    Log only for the first n times.

    Parameters
    ----------
    lvl : int
        the logging level
    msg : str
        the message to log
    n : int
        the number of times to log
    name : str
        name of the logger to use. Will use the caller's module by default.
    key : str
        the key to use for counting. If "caller", will use the caller's module name.
        If "message", will use the message string.
    """
    if isinstance(key, str):
        key = (key,)
    assert len(key) > 0

    caller_module, caller_key = caller_identity()
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (message,)

    _log_counter[hash_key] += 1
    if _log_counter[hash_key] <= n:
        logging.getLogger(name or caller_module).log(level, message)
