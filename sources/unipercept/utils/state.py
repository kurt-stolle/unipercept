from __future__ import annotations

import accelerate
import functools
import dataclasses as D

@D.dataclass(kw_only=True, slots=True)
class _ProcessStateManager:
    interactive: bool = False

def _accelerate_state():
    return accelerate.PartialState()

@functools.lru_cache(maxsize=None)
def _unipercept_state():
    return _ProcessStateManager()

def get_interactive():
    return _unipercept_state().interactive    


def get_process_index(local=False):
    return _accelerate_state().local_process_index if local else _accelerate_state().process_index


def get_process_count() -> int:
    return _accelerate_state().num_processes


def check_main_process(local=False):
    return _accelerate_state().is_local_main_process if local else _accelerate_state().is_main_process


def check_debug_enabled():
    return _accelerate_state().debug


def barrier(*args):
    return _accelerate_state().wait_for_everyone()


_PROXY_ACCELERATE_ARRTS = {
    "main_process_first",
    "local_main_process_first",
    "on_main_process",
    "on_process",
    "on_last_process",
    "print",
}


def __getattr__(name: str):
    if name in _PROXY_ACCELERATE_ARRTS:
        return getattr(_accelerate_state(), name)
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
