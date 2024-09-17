from __future__ import annotations

import typing as T


class Interval(T.NamedTuple):
    """
    The engine runs on intervals of steps, which can be defined in terms of epochs.

    Traditionally the epoch is defined as the amount of steps to be trained such
    that the model has 'seen' the full dataset. This is however not always the case or
    results in vague definitions, e.g. in the case of random clipping or
    infinite data sources.

    We recommend interpreting the ``steps_per_epoch`` as a
    hyperparameter that defines when the model has seen the scope of the dataset,
    i.e. all classes have been seen in most of the possible contexts.
    """

    amount: int
    unit: T.Literal["steps", "epochs"]

    def get_steps(self, steps_per_epoch: int) -> int:
        if self.unit == "steps":
            return self.amount
        if self.unit == "epochs":
            return self.amount * steps_per_epoch

        msg = f"Unknown unit {self.unit}"
        raise ValueError(msg)

    def get_epochs(self, steps_per_epoch: int) -> float:
        if self.unit == "steps":
            return self.amount // steps_per_epoch
        if self.unit == "epochs":
            return self.amount

        msg = f"Unknown unit {self.unit}"
        raise ValueError(msg)
