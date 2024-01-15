"""
Integrations with third-party libraries.
"""
from __future__ import annotations

from unipercept.utils.module import lazy_module_factory

__all__ = []
__getattr__, __dir__ = lazy_module_factory(__name__, ["slurm_integration", "wandb_integration"])

del lazy_module_factory
