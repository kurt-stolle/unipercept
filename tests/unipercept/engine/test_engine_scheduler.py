from __future__ import annotations

import pytest
import torch.optim
from timm.scheduler.cosine_lr import CosineLRScheduler as TimmCosineLRScheduler
from timm.scheduler.scheduler import Scheduler as TimmScheduler
from unipercept.engine import SchedType, SchedulerFactory


@pytest.mark.parametrize("sched_type", [t for t in SchedType])
def test_scheduler_factory(model, sched_type):
    sched_lazy = SchedulerFactory(sched_type)
    params = model.parameters()
    optim = torch.optim.SGD(params, lr=0.1)
    scheduler, epochs = sched_lazy(optim, 300)
    assert (
        issubclass(type(scheduler), TimmScheduler)
        or isinstance(scheduler, TimmScheduler)
    ), f"Expected {TimmScheduler}, got {type(scheduler)}, which is not a subclass of {TimmScheduler}"
    assert isinstance(epochs, int), f"Expected int, got {type(epochs)}"
    assert epochs > 0, f"Expected epochs > 0, got {epochs}"


def test_scheduler_wraped(model):
    params = model.parameters()
    optim = torch.optim.SGD(params, lr=0.1)
    sched_lazy = SchedulerFactory(TimmCosineLRScheduler, lr_min=0.0)
    scheduler = sched_lazy(optim, 300)
    assert (
        issubclass(type(scheduler), TimmScheduler)
        or isinstance(scheduler, TimmScheduler)
    ), f"Expected {TimmScheduler}, got {type(scheduler)}, which is not a subclass of {TimmScheduler}"
