"""
Logging utilities. Package name intentionally set to be sufficiently unique.
"""

from __future__ import annotations

import logging
from typing import Optional

__all__ = ["get_logger", "LOG_LEVELS"]

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "passive": -1,
}


def get_logger(name: Optional[str] = None, process_index: int = 0, abbrev_name: Optional[str] = None, **kwargs):
    """
    Wraps the logger provided by ``detectron2.utils.logger``.

    Automatically assigns a name based on the calling module.

    If the name starts with ``unipercept``, the first part of the name will be abbreviated to ``up``.
    """
    from detectron2.utils.logger import setup_logger

    from .inspect import calling_module_name

    if name is None:
        name = calling_module_name(left=1)

    if abbrev_name is None:
        abbrev_name = name.replace("unipercept", "up")

    logger = setup_logger("unipercept", process_index, name=name, abbrev_name=abbrev_name, **kwargs)

    return logger
