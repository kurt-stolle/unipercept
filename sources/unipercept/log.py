"""
Canonical logger used throughout the project.

This module follows the same semantics as defined in the PyTorch logging.

See Also
--------
- `PyTorch Logging <https://pytorch.org/docs/stable/_modules/torch/_logging.html>`_
- `Design Document <https://docs.google.com/document/d/1ZRfTWKa8eaPq1AxaiHrq4ASTPouzzlPiuquSBEJYwS8/edit>`_
"""

from __future__ import annotations

import atexit
import collections
import enum as E
import functools
import io
import logging
import os
import sys
import time
import typing as T
import warnings
from collections import Counter
from typing import override

import pandas as pd
from tabulate import tabulate
from termcolor import colored

import unipercept.utils.inspect as inspect_utils
from unipercept import file_io

__all__ = []

Logger: T.TypeAlias = logging.Logger

# Logging levels are defined by integer values, this mapping makes it easier to use strings, e.g. in a CLI setting.
LOG_LEVELS: T.Final[T.OrderedDict[str, int]] = T.OrderedDict(
    {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
)


# Default logging level, can be overridden by setting the UN_LOG_LEVEL environment variable.
@functools.cache
def _get_logging_level_map() -> collections.defaultdict[str, str]:
    """
    Returns a mapping of (logger name) -> (logging level).

    By default, all loggers are set to the level of the root logger, which has
    a default value set by the environment variable ``UP_LOGS_LEVEL``, by default
    ``logging.INFO``.

    Individual modules' log levels can be overridden by setting the environment
    variable ``UP_LOGS``.
    The environment variable should be a comma-separated list of logger names and
    levels. We allow two formats for each entry:

        1. Explicitly pass the module name and log level as ``module name:log level``.
           Example: ``module1:info,module2:debug``.
        2. Prepend the module name with one or multiple `+` or `-` signs to respectively
           increasee verbosity. The adjustment is centered around the default log level
           as specified by the ``UP_LOGS_LEVEL`` environment variable (see above).
           Example: ``+module1,-module2,++module3``.

    Entries are parsed in the order above, so combinations are allowed, though
    this is not recommended for readability.
    """

    level_options = list(LOG_LEVELS.keys())
    level_base = os.getenv("UP_LOGS_LEVEL", "info").lower()
    result = collections.defaultdict(lambda: LOG_LEVELS[level_base])

    for spec in os.getenv("UP_LOGS", "").lower().split(","):
        spec = spec.strip()
        if not any(c in spec for c in "+-:"):
            continue
        spec, level = spec.split(":") if ":" in spec else (spec, level_base)
        if level not in level_options:
            msg = f"Unknown log level: {level}. Options are: {', '.join(level_options)}"
            raise ValueError(msg)
        module_name = spec.lstrip("+-")
        spec_adjust = spec[: len(spec) - len(module_name)]
        rel_amount = int(spec_adjust.count("+")) - int(spec_adjust.count("-"))
        cur_level = level_options.index(level)
        new_level = min(len(LOG_LEVELS), max(0, cur_level + rel_amount))

        result[module_name] = level_options[new_level]

    return result


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

    caller_module, key = inspect_utils.caller_identity()
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
    caller_module, key = inspect_utils.caller_identity()
    last_logged = _log_timers.get(key)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        logging.getLogger(name or caller_module).log(_get_level(level), message)
        _log_timers[key] = current_time


class TableFormat(E.StrEnum):
    LONG = E.auto()
    WIDE = E.auto()
    AUTO = E.auto()


def create_table(
    mapping: T.Mapping[T.Any, T.Any] | pd.DataFrame,
    format: TableFormat | T.Literal["long", "wide", "auto"] = TableFormat.AUTO,
    *,
    style: str = "rounded_outline",
    max_depth: int = 5,
    max_width: int = 120,
    index: bool = False,
    _depth: int = 0,
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
    from pprint import pformat

    tabulate_kwargs: dict[str, T.Any] = {
        "tablefmt": style,
        "floatfmt": ".3f",
        "stralign": "left",
        "numalign": "right",
    }

    # Handling of various input types that need to be converted to a dictionary or
    # can be handled directly.
    if isinstance(mapping, pd.DataFrame):
        # In case of a Pandas DataFrame, we first try to convert it to a table by using
        # the built-in `to_markdown` method. Note that this calls `tabulate` under
        # the hood, so we do not actually get Markdown back, but rather the desired
        # table as a string following our own specification.
        try:
            result = mapping.to_markdown(index=index, **tabulate_kwargs)
        except Exception as e:
            warnings.warn(
                "Failed to convert DataFrame to table: " + str(e), stacklevel=2
            )
            result = None
        # Check whether the conversion was successful
        if isinstance(result, str):
            return result
        # Fallback to converting the dataframe to a dict in Pandas versions that do
        # not support passing Tabulate keyword-arguments
        mapping = mapping.to_dict(orient="list")
    elif hasattr(mapping, "_asdict") and callable(mapping._asdict):
        # Support for namedtuples, which do not support instance checks.
        mapping = mapping._asdict()

    # Determine the format of the table
    if format == TableFormat.AUTO:
        if len(mapping) <= 5 and _depth <= 1:
            format = TableFormat.WIDE
        else:
            format = TableFormat.LONG

    if format == TableFormat.WIDE:
        headers = list(mapping.keys())
        # Create a wide table, i.e. where each key is a header and the values are under
        # the corresponding header. If the value is a non-sequence, it will be displayed as-is
        data = []
        for v in mapping.values():
            if isinstance(v, dict):
                v = create_table(
                    v,
                    format="auto",
                    style=style,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
            if isinstance(v, str):
                v = v.split("\n")
            elif isinstance(v, T.Sequence):
                v = [pformat(v_item) for v_item in v]
            else:
                v = [v]
            data.append(v)

        col_lens = [len(v) for v in data]
        if len(col_lens) > 1:
            pad_to = max(col_lens)
        else:
            pad_to = 1

        for v in data:
            v.extend([""] * (pad_to - len(v)))
        # Transpose the data to make it wide
        data = list(zip(*data, strict=False))
    elif format == TableFormat.LONG:
        data = []
        for k, v in mapping.items():
            if isinstance(v, dict) and _depth < max_depth:
                for linenum, line in enumerate(
                    create_table(
                        v,
                        format="auto",
                        style=style,
                        max_depth=max_depth,
                        _depth=_depth + 1,
                    ).split("\n")
                ):
                    data.append((k if linenum == 0 else "", line))
                continue
            if not isinstance(v, str):
                try:
                    v = pformat(v)
                except Exception:  # noqa: PIE786
                    v = str(v)

            # Truncate long lines
            v_lines = v.split("\n")
            for v_i in range(len(v_lines)):
                v_line = v_lines[v_i]
                if len(v_line) > max_width:
                    v_line = v_line[: max_width - 3] + "..."

                v_line = v_line.replace("\t", "  ")
                if (
                    v_line[0] == " "
                ):  # prevent tabulate from removing leading whitespace
                    v_line = "␣" + v_line[1:]
                v_lines[v_i] = v_line

            # Put the first line in the same row as the key
            data.append((k, v_lines[0]))
            for v_line in v_lines[1:]:
                data.append(("", v_line))
        # Create a long table, i.e. with a 'key' and 'value' header
        headers = ["Key", "Value"] if _depth == 0 else ()
    else:
        msg = f"Unknown table format: {format}"
        raise ValueError(msg)

    return tabulate(data, headers=headers, showindex=index, **tabulate_kwargs)


def get_logger(name: str | None = None, **kwargs: T.Any) -> logging.Logger:
    """
    Get a logger instance, where the name is automatically set to the calling module name.
    """

    return _get_handler(
        _canonicalize_name(name or inspect_utils.calling_module_name(left=1)),
        **kwargs,
    )


@functools.cache
def _canonicalize_name(name: str) -> str:
    """
    Canonicalizes a logger name into a full module name without private submodules.
    """
    result = []
    for name in name.split(" "):
        if name.endswith(".py"):
            file = file_io.Path(name)
            root = file_io.Path(__file__).parent
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
            name = "<unknown>"

        result.append(name)
    return " ".join(result)


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


@functools.cache
def _get_handler(
    name: str,
    /,
    *,
    output=None,
    color=True,
    propagate: bool = False,
    stdout: bool = True,
    level: str | int | None = None,
):
    """
    Get a logger with a default verbose formatter, cached to ensure that the same logger handler is shared across
    all the sites that call this function.
    """
    from unipercept.state import get_process_index

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = propagate

    # Determine the full name and the short name
    root_name, *sub_name = name.split(".", 1)
    module_name, *artifact_name = name.split(" ", 1)

    # Translate the logging level
    if level is None:
        level = _get_logging_level_map()[module_name]
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


@functools.cache
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


@functools.cache
def _get_stream(filename) -> io.IOBase:
    """
    Taken from the `detectron` implementation.
    """
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

    caller_module, caller_key = inspect_utils.caller_identity()
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (message,)

    _log_counter[hash_key] += 1
    if _log_counter[hash_key] <= n:
        logging.getLogger(name or caller_module).log(level, message)


########################
# Attribute forwarding #
########################


def __getattr__(name: str) -> T.Any:
    """
    Forward all unknown attributes to the logging module.
    """
    if name == "logger" or name in (
        "debug",
        "info",
        "warning",
        "error",
        "critical",
        "exception",
    ):
        logger = get_logger(inspect_utils.calling_module_name(frames=0, strict=False))
        if name == "logger":
            return logger
        return getattr(logger, name)

    msg = f"Module {__name__} has no attribute {name}"
    raise AttributeError(msg)


if T.TYPE_CHECKING:
    logger: T.Final[logging.Logger] = logging.Logger("<caller>")
    debug = logger.debug
    info = logger.info
    warning = logger.warning
    error = logger.error
    critical = logger.critical
    exception = logger.exception
