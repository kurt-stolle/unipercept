from __future__ import annotations

import typing as T
from dataclasses import dataclass

if T.TYPE_CHECKING:
    from evaluators import Evaluator

    from unipercept.data import DataLoaderFactory

__all__ = ["DataConfig"]


@dataclass
class DataConfig:
    loaders: T.Dict[str, DataLoaderFactory]
    evaluators: T.Optional[T.Sequence[Evaluator]] = None
