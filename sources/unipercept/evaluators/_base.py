"""
Implements the baseclass for evaluators
"""

from __future__ import annotations

import abc
import contextlib
import enum as E
import dataclasses as D
import typing as T

import pandas as pd
import torch
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDictBase

from unipercept.log import get_logger
from unipercept.state import check_main_process, get_interactive

if T.TYPE_CHECKING:
    from unipercept.model import InputData
    from tensordict import TensorDictBase
    from unipercept.data.sets import Metadata

__all__ = ["Evaluator", "PlotMode"]


_logger = get_logger(__name__)


class PlotMode(E.StrEnum):
    ALWAYS = E.auto()
    ONCE = E.auto()
    NEVER = E.auto()


class StoragePrefix(E.StrEnum):
    """
    Simple enum for common key prefixes in the storage. Use of this enum is optional,
    but it can be helpful in the canonicalization of evaluators.
    """

    TRUE = E.auto()
    PRED = E.auto()
    VALID = E.auto()


class EvaluatorComputeKWArgs(T.TypedDict):
    """
    A type for the keyword arguments passed to the ``compute`` method of an evaluator.
    """

    device: torch.types.Device
    path: str


@D.dataclass(kw_only=True)
class Evaluator(metaclass=abc.ABCMeta):
    """
    Implements an evaluator for a given task.
    """

    @classmethod
    def from_metadata(cls, name: str, **kwargs) -> T.Self:
        from unipercept import get_info

        return cls(info=get_info(name), **kwargs)

    def __new__(cls, *args, **kwargs):
        if cls is Evaluator:
            msg = "Cannot instantiate base class Evaluator directly."
            raise RuntimeError(msg)
        return super().__new__(cls)

    info: Metadata = D.field(repr=False)

    show_progress: bool = D.field(
        default_factory=get_interactive,
        metadata={
            "help": "Show progress bar",
        },
    )
    show_summary: bool = True
    show_details: bool = False
    prefix: str | None = D.field(
        default=None,
        metadata={
            "help": "Prefix for storage keys, used to avoid conflicts between different evaluators",
        },
    )
    pair_index: int = D.field(
        default=-1,
        metadata={
            "help": (
                "Index of the temporal pair dimension in the batch items. "
                "For example, if a batch item has images with (B, T, C, H, W) shape, "
                "then the default pair index `-1` indices that the prediction is "
                "temporally aligned to the last (-1th) image in the T-dimension."
            )
        },
    )

    @abc.abstractmethod
    def update(
        self,
        storage: TensorDictBase,  # noqa: U100
        inputs: InputData,  # noqa: U100
        outputs: TensorDictBase,  # noqa: U100
    ) -> None:
        pass

    @abc.abstractmethod
    def compute(
        self,
        storage: TensorDictBase,  # noqa: U100
        **kwargs: T.Unpack[EvaluatorComputeKWArgs],  # noqa: U100
    ) -> dict[str, int | float | str | bool | dict]:
        return {}

    @abc.abstractmethod
    def plot(
        self,
        storage: TensorDictBase,  # noqa: U100
    ) -> dict[str, pil_image.Image]:
        return {}

    def _show_table(self, msg: str, tab: pd.DataFrame) -> None:
        from unipercept.log import create_table

        # tab_fmt = tab.to_markdown(index=False)

        tab_fmt = create_table(tab.to_dict(orient="list"))

        _logger.info("%s:\n%s", msg, tab_fmt)

    def _progress_bar(self, *args, **kwargs):
        from tqdm import tqdm

        return tqdm(
            *args,
            dynamic_ncols=True,
            disable=not check_main_process(local=True) or not self.show_progress,
            **kwargs,
        )

    def get_storage_key(self, key: str, prefix: StoragePrefix | str) -> str:
        if isinstance(prefix, StoragePrefix):
            kind = prefix.value
        else:
            kind = str(prefix)
        if self.prefix is not None:
            key = f"{self.prefix}_{key}"
        return "_".join([kind, key])
