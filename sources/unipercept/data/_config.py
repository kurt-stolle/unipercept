import typing as T
from dataclasses import dataclass

from ._loader import DataLoaderFactory


@dataclass
class DataConfig:
    loaders: T.Dict[str, DataLoaderFactory]
    evaluator: T.Optional[T.Any] = None
