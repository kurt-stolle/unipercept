from __future__ import annotations

from datetime import datetime

__all__ = ["get_timestamp"]


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
