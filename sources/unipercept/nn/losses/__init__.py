"""
This module hosts various losses for perception tasks.
"""

from __future__ import annotations

from unipercept.utils.module import lazy_module_factory

__all__ = [
    "chamfer",
    "contrastive",
    "depth",
    "focal",
    "functional",
    "guided",
    "image",
    "mixins",
    "panoptic",
]
__getattr__, __dir__ = lazy_module_factory(__name__, __all__)

del lazy_module_factory
