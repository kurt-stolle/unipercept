"""
Tests ``unipercept.evaluators.dvps`` against the reference implementation.

Results to the reference implementation were computed ahead-of-time using the
steps listed in at the implementation <https://github.com/joe-siyuan-qiao/ViP-DeepLab>.
"""

from __future__ import annotations

import importlib
import zipfile
from pathlib import Path

import pytest
from tensordict import TensorDictBase
import safetensors.torch

from unipercept import file_io
from unipercept.evaluators import DVPSEvaluator, DVPSWriter
from unipercept.engine.writer import MemmapTensorDictWriter

@pytest.fixture
def cityscapes_storage(tmp_path) -> TensorDictBase:
    """
    Write depth and panoptic predictions to a temporary directory and return the path
    to that directory
    """
    tmp_path = tmp_path / "cityscapes-results"
    tmp_path.mkdir()

    zip_path = file_io.Path("https://mps-gpu-02.ele.tue.nl/unipercept/testing/dvps/cityscapes/raw-val.zip")
    out_path = tmp_path / "raw"
    with zipfile.ZipFile(zip_path, "r") as fh:
        fh.extractall(out_path)
    assert out_path.is_dir()

    wr = MemmapTensorDictWriter(out_path / "writer")
    
    for path in out_path.glob("*.safetensors"):
        raw_dict = safetensors.torch.load_file(path)

        # Write to writer

    wr.close()

    yield wr

    tmp_path.rmdir()
    zip_path.unlink()




    


def test_dvpq_evaluator_cityscapes():
    
