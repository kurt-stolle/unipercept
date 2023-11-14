"""UniPercept: A Python package for the analysis of perceptual data and the training of deep neural networks in PyTorch."""
from __future__ import annotations

from unicore.utils.module import lazy_module_factory

from ._api import *
from ._patch_tensordict_pytree import *

__version__ = "3.2.2"
__all__ = [
    # Submodules
    "data",
    "model",
    "nn",
    "render",
    "trainer",
    "utils",
    "evaluators",
    # API exports (see: ./_api.py)
    "read_config",
    "load_checkpoint",
    "create_model",
    "prepare_dataset",
    "create_inputs",
    "prepare_images",
    "read_image",
]
__getattr__, __dir__ = lazy_module_factory(__name__, __all__)

del lazy_module_factory
