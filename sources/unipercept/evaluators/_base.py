"""
Implements the baseclass for evaluators
"""
from __future__ import annotations

import abc
import contextlib
import dataclasses as D
import enum
import typing as T

import pandas as pd
import torch
import torch.types
from PIL import Image as pil_image
from tensordict import TensorDictBase

from unipercept.log import get_logger
from unipercept.state import check_main_process

if T.TYPE_CHECKING:
    from ..model import TensorDictBase

__all__ = ["Evaluator", "PlotMode"]


_logger = get_logger(__name__)


class PlotMode(enum.Enum):
    ALWAYS = enum.auto()
    ONCE = enum.auto()
    NEVER = enum.auto()


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

    show_progress: bool = D.field(default=True, metadata={"help": "Show progress bar"})

    def update(
        self,
        storage: TensorDictBase,  # noqa: U100
        inputs: TensorDictBase,  # noqa: U100
        outputs: TensorDictBase,  # noqa: U100
    ) -> None:
        ...

    @abc.abstractmethod
    def compute(
        self,
        storage: TensorDictBase,  # noqa: U100
        **kwargs: T.Unpack[EvaluatorComputeKWArgs],  # noqa: U100
    ) -> dict[str, int | float | str | bool | dict]:
        ...

    @abc.abstractmethod
    def plot(
        self,
        storage: TensorDictBase,  # noqa: U100
    ) -> dict[str, pil_image.Image]:
        ...

    def _show_table(self, msg: str, tab: pd.DataFrame) -> None:
        tab_fmt = tab.to_markdown(index=False)
        _logger.info("%s:\n%s", msg, tab_fmt)

    def _progress_bar(self, *args, **kwargs):
        from tqdm import tqdm

        return tqdm(
            *args,
            dynamic_ncols=True,
            disable=check_main_process(local=True) or not self.show_progress,
            **kwargs,
        )
