"""
Seed utilities for reproducible behavior during training.

Based on PyTorch Lightning.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_worker", "set_seed"]


def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def set_seed(seed: int, fully_deterministic: bool = False):
    if fully_deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
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
