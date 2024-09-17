r"""
Implements an evaluator that wraps the ``torchmetrics`` library.

See Also
--------

- `Torchmetrics <https://lightning.ai/docs/torchmetrics/stable/>`_.
"""

from __future__ import annotations

try:
    import torchmetrics
except ImportError:
    torchmetrics = None

from ._base import Evaluator


def check_torchmetrics_available() -> bool:
    return torchmetrics is not None


class TorchmetricsEvaluator(Evaluator):
    def __post_init__(self):
        if not check_torchmetrics_available():
            raise ImportError("torchmetrics is not available.")


def __getattr__(name):
    raise NotImplementedError("This module is not implemented yet.")
