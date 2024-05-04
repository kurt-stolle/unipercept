"""
Tests ``unipercept.evaluators.dvps`` against the reference implementation.

Results to the reference implementation were computed ahead-of-time using the
steps listed in at the implementation <https://github.com/joe-siyuan-qiao/ViP-DeepLab>.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from tensordict import TensorDictBase

from unipercept import file_io
from unipercept.evaluators import DVPSEvaluator, DVPSWriter


@pytest.fixture
def cityscapes_storage() -> TensorDictBase:
    """
    Write depth and panoptic predictions to a temporary directory and return the path
    to that directory
    """
    


def test_dvpq_evaluator_cityscapes():
    
