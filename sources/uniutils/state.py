from __future__ import annotations

import dataclasses
import typing as T
from threading import RLock

import accelerate

_distributed = accelerate.PartialState()


def get_process_index(*args, local=False):
    return _distributed.local_process_index if local else _distributed.process_index


def get_process_count(*args) -> int:
    return _distributed.num_processes


def check_main_process(*args, local=False):
    return _distributed.is_local_main_process if local else _distributed.is_main_process


def check_debug_enabled(*args):
    return _distributed.debug


def barrier(*args):
    return _distributed.wait_for_everyone()


_PROXY_ATTRS = {
    "main_process_first",
    "local_main_process_first",
    "on_main_process",
    "on_process",
    "on_last_process",
    "print",
}


def __getattr__(name: str):
    if name in _PROXY_ATTRS:
        return getattr(_distributed, name)
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
