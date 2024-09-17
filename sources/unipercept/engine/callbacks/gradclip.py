import enum as E
import typing as T

import torch
from torch import nn

from .._params import EngineParams
from ..accelerate import Accelerator
from ._base import CallbackDispatcher, Signal, State


class GradientClippingMode(E.StrEnum):
    """
    Enum for the gradient clipping mode.
    """

    GLOBAL = "global"
    PARAMETER = "parameter"


class GradientClippingCallback(CallbackDispatcher):
    """
    A ``Callback`` that handles gradient clipping during training.
    """

    def __init__(
        self,
        max_norm: float | None = None,
        max_value: float | None = None,
        norm_tracker: nn.Module | None = None,
        mode: GradientClippingMode = GradientClippingMode.GLOBAL,
    ):
        """
        Parameters
        ----------
        max_norm:
            The maximum total norm of all gradients.
        max_value:
            The maximum absolute value of any individual gradient.
        norm_tracker:
            The tracker module to use for tracking the total norm of the gradients.
            If None, the gradient is clipped based only by the ``max_norm`` value.
        mode:
            The mode of the gradient clipping. See :class:`GradientClippingMode`.
        """
        self.max_norm = max_norm
        self.max_value = max_value
        self.norm_tracker = norm_tracker
        self.mode = mode
        self.total_norm: Tensor | None = None
        self.step_counter: Tensor | None = None

        assert (
            self.max_norm is None or self.max_norm >= 0
        ), "max_norm must be non-negative or disabled"
        assert (
            self.max_value is None or self.max_value >= 0
        ), "max_value must be non-negative or disabled"
        assert (self.norm_tracker is None) or (
            self.norm_tracker is not None and self.max_norm is not None
        ), "max_norm must be defined when using a tracker"

    @T.override
    def on_accelerator_setup(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        accelerator: Accelerator,
        **kwargs,
    ):
        if self.norm_tracker is not None:
            accelerator.register_for_checkpointing(self.norm_tracker)
        if self.norm_tracker is not None or self.max_norm is not None:
            self.total_norm = torch.tensor(
                0.0, device=accelerator.device, dtype=torch.float32, requires_grad=False
            )
        self.step_counter = torch.tensor(
            0, device=accelerator.device, dtype=torch.int32, requires_grad=False
        )

    @T.override
    def on_train_begin(
        self, params: EngineParams, state: State, control: Signal, **kwargs
    ):
        assert self.step_counter is not None
        self.step_counter.zero_()

        if self.total_norm is not None:
            self.total_norm.zero_()

        if self.norm_tracker is not None:
            self.norm_tracker.reset()

    @T.override
    def on_log(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        logs: dict[str, T.Any],
        **kwargs,
    ):
        assert self.step_counter is not None
        if self.step_counter == 0:
            return

        if self.total_norm is not None:
            logs["optimizer/total_norm"] = (self.total_norm / self.step_counter).item()
            self.total_norm.zero_()

        if self.norm_tracker is not None:
            smooth_norm = self.norm_tracker.observe().item()
            logs["optimizer/smooth_norm"] = smooth_norm

        self.step_counter.zero_()

    @T.override
    def on_train_substep_end(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        step_last_logged: int,
        **kwargs,
    ):
        if self.total_norm is not None:
            self.total_norm += self.total_norm / (1 + state.step - step_last_logged)

    @T.override
    def on_train_gradients(
        self,
        params: EngineParams,
        state: State,
        control: Signal,
        *,
        model: nn.Module,
        **kwargs,
    ):
        assert self.step_counter is not None
        if self.mode != GradientClippingMode.GLOBAL:
            msg = f"Gradient clipping mode {self.mode} is not implemented."
            raise NotImplementedError(msg)

        model_params = list(model.parameters())
        for p in model_params:
            if p is None or p.grad is None:
                continue
            torch.nan_to_num(p.grad, nan=0.0, posinf=1e8, neginf=-1e8, out=p.grad)

        self.step_counter += 1

        # Clip gradients by value
        if self.max_value is not None:
            nn.utils.clip_grad_value_(model_params, self.max_value)

        # Clip gradients by norm
        if self.max_norm is not None:
            assert self.total_norm is not None

            max_norm: float
            # Read the max norm value (from smoother or directly from params)
            if self.norm_tracker is not None:
                max_norm_obs = self.norm_tracker.observe()
                if not torch.isfinite(max_norm_obs):
                    max_norm = self.max_norm
                else:
                    max_norm = max_norm_obs.item()
                    max_norm = min(max_norm, self.max_norm)
            else:
                max_norm = self.max_norm

            # Apply gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(
                model_params, max_norm, norm_type=2
            )
            total_norm = total_norm.detach()

            # Smooth the gradient norm
            if self.norm_tracker is not None and torch.isfinite(total_norm):
                self.norm_tracker(total_norm)

            # Update the total norm
            self.total_norm.add_(total_norm)
