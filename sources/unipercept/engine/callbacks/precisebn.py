import typing as T

import torch.utils.data as torchdata
from torch import nn

from unipercept.log import logger
from unipercept.types import Tensor

from .._params import EngineParams, Interval
from ._base import CallbackDispatcher, Signal, State


class PreciseBatchNormCallback(CallbackDispatcher):
    """
    Runs the precise batch norm algorithm to convergence.

    See Also
    --------
    - https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/precise_bn.py
    """

    __slots__ = ("interval", "iterations", "show_progress")

    def __init__(self, interval: Interval, iterations: int, show_progress=True):
        """
        Parameters
        ----------
        interval:
            The interval at which to run the precise batch norm algorithm.
        iterations:
            The number of iterations to run the precise batch norm algorithm.
        show_progress:
            Whether to show the progress bar.
        """

        self.interval = interval
        self.iterations = iterations
        self.show_progress = show_progress

    def compute_precise_batchnorm(self, model: nn.Module, loader: torchdata.DataLoader):
        from fvcore.nn.precise_bn import update_bn_stats

        logger.info("Computing precise batch norm statistics...")
        update_bn_stats(model, loader, self.iterations, progress=self.show_progress)

    @T.override
    def on_train_step_end(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        losses: dict[str, Tensor],
        model: ModelBase,
        loader: DataLoader,
        **kwargs,
    ):
        if self.interval.unit != "steps":
            return
        if state.step % self.interval.amount > 0:
            return

        self.compute_precise_batchnorm(model, loader)

    @T.override
    def on_train_epoch_end(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        model: ModelBase,
        loader: DataLoader,
        **kwargs,
    ):
        if self.interval.unit != "epochs":
            return

        if round(state.epoch) % self.interval.amount > 0:
            return
        self.compute_precise_batchnorm(model, loader)
