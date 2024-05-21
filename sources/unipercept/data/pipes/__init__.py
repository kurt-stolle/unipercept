"""
Implements various ``torchdata.datapipe`` pipelines.
"""

from __future__ import annotations

import torch

torch.utils.data.datapipes.utils.common.DILL_AVAILABLE = (
    torch.utils._import_utils.dill_available()
)
from ._cache import *
from ._io_wrap import *
from ._join import *
from ._pattern import *
from ._pil import *
