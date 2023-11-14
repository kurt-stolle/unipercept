"""
Seed utilities for reproducible behavior during training.

Based on PyTorch Lightning.
"""


from __future__ import annotations

__all__ = ["seed_worker", "enable_full_determinism", "set_seed"]


def seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)


def enable_full_determinism(seed: int, warn_only: bool = False):
    import os

    import numpy as np
    import torch

    set_seed(seed)

    # Enable PyTorch deterministic mode. This potentially requires either the environment
    # variable 'CUDA_LAUNCH_BLOCKING' or 'CUBLAS_WORKSPACE_CONFIG' to be set,
    # depending on the CUDA version, so we set them both here
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
