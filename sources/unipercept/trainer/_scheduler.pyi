from __future__ import annotations

import enum
import functools
import typing as T

from timm.scheduler.scheduler import Scheduler
from torch.optim import Optimizer

_S = T.TypeVar("_S", bound=Scheduler)
_P = T.ParamSpec("_P")

Optimizer = Optimizer
Scheduler = Scheduler
SchedulerAndEpochs: T.TypeAlias = tuple[_S, int]

__all__ = ["SchedType", "create_scheduler", "SchedulerFactory"]
__dir__ = ["SchedType", "create_scheduler", "SchedulerFactory"]

class SchedType(enum.Enum):
    COSINE = enum.auto()
    TANH = enum.auto()
    STEP = enum.auto()
    MULTISTEP = enum.auto()
    PLATEAU = enum.auto()
    POLY = enum.auto()

class SchedulerFactory:
    @T.overload
    def __init__(
        self,
        scd: str | SchedType = SchedType.POLY,
        *,
        decay_epochs: int = 90,
        decay_milestones: T.Sequence[int] = (90, 180, 270),
        cooldown_epochs: int = 0,
        patience_epochs: int = 10,
        decay_rate: float = 0.1,
        min_lr: float = 0,
        warmup_lr: float = 1e-5,
        warmup_epochs: int = 0,
        warmup_prefix: bool = False,
        noise: float | T.Sequence[float] | None = None,
        noise_pct: float = 0.67,
        noise_std: float = 1.0,
        noise_seed: int = 42,
        cycle_mul: float = 1.0,
        cycle_decay: float = 0.1,
        cycle_limit: int = 1,
        k_decay: float = 1.0,
        plateau_mode: str = "max",
        step_on_epochs: bool = True,
        updates_per_epoch: int = 0,
    ): ...
    @T.overload
    def __init__(
        self,
        scd: T.Callable[T.Concatenate[T.Any, _P], _S],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ): ...
    def __init__(
        self,
        scd,
        **kwargs,
    ): ...
    def __call__(self, optimizer: Optimizer, num_epochs: int) -> SchedulerAndEpochs: ...

def create_scheduler(
    scd: SchedType | str,
    optimizer: Optimizer,
    *,
    t_initial: int,
    min_lr: float,
    step_on_epochs: bool,
    updates_per_epoch: int,
    num_epochs: int,
    decay_t: int,
    decay_milestones: T.Sequence[int],
    decay_rate: float,
    cooldown_t: int,
    patience_epochs: int,
    k_decay: float,
    plateau_mode: str,
    cycle_args: dict,
    noise_args: dict,
    warmup_args: dict,
) -> Scheduler: ...
