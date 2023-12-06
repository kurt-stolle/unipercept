from __future__ import annotations

import dataclasses as D
import functools

import accelerate


@D.dataclass(kw_only=True, slots=True)
class _ProcessStateManager:
    interactive: bool = False


_state_xlr = accelerate.PartialState()
_state_up = _ProcessStateManager()


def get_interactive():
    return _state_up.interactive


def get_process_index(local=False):
    return _state_xlr.local_process_index if local else _state_xlr.process_index


def get_process_count() -> int:
    return _state_xlr.num_processes


def check_main_process(local=False):
    return _state_xlr.is_local_main_process if local else _state_xlr.is_main_process


def check_debug_enabled():
    return _state_xlr.debug


def barrier(*args):
    return _state_xlr.wait_for_everyone()


main_process_first = _state_xlr.main_process_first
local_main_process_first = _state_xlr.local_main_process_first
print = _state_xlr.print


def on_process():
    return _state_xlr.on_process


def on_last_process():
    return _state_xlr.on_last_process


def on_main_process():
    return _state_xlr.on_main_process
