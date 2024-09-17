"""
Seed utilities for reproducible behavior during training.

Based on PyTorch Lightning.
"""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch

__all__ = ["seed_worker", "set_seed"]


DEFAULT_SEED = 1958


def set_seed(seed: int = DEFAULT_SEED, fully_deterministic: bool = False):
    """
    Set seed for reproducible behavior.

    Parameters
    ----------
    seed : int
        Seed value to set. By default, 1958.
    fully_deterministic : bool
        Whether to set the environment to fully deterministic. By default, False.
        This should only be used for debugging and testing, as it can significantly
        slow down training at little to no benefit.
    """
    if fully_deterministic:
        warnings.warn(
            "Fully deterministic mode is enabled. This will induce significant latency.",
            stacklevel=0,
        )

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)
