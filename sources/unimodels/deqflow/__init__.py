"""
Implements the DEQFlow network for optical flow estimation.

Paper: https://arxiv.org/pdf/2204.08442.pdf

We use an adapted version of the authors' code.

Adapted version: https://github.com/kurt-stolle/deq-flow
Original version: https://github.com/locuslab/deq-flow
"""

from __future__ import annotations

from ._model import *
