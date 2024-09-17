r"""
Defines models using a standardized template with consistent I/O and configuration handling.
"""

from __future__ import annotations

from unipercept.utils.module import lazy_module_factory

# Always import privates
from ._base import *
from ._io import *

# Other modules are imported lazily
__getattr__, __dir__ = lazy_module_factory(__name__, ["toys"])

del lazy_module_factory
