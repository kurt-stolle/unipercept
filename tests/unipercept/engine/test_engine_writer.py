from __future__ import annotations

import shutil
import typing as T

import pytest
import torch
import typing_extensions as TX
from tensordict import TensorDict
from pathlib import Path

from unipercept.engine.writer import PersistentTensordictWriter, MemmapTensordictWriter


NUM_SAMPLES = 32


def _run_write_read_benchmark(writer: T.Union[MemmapTensordictWriter, PersistentTensordictWriter]):
    for _ in range(NUM_SAMPLES):
        writer.add(TensorDict({"a": torch.randn(1, 64, 120), "z": torch.ones(1,1)}, batch_size=[1]))
    writer.flush()

    td = writer.tensordict

    for i in range(NUM_SAMPLES):
        item = td.get_at("a", i)
        assert item.shape == ( 64, 120)

    writer.close()
    

def test_memmap_writer(benchmark, tmp_path: Path):
    @benchmark
    def write_then_read():
        path = tmp_path / "memmap_writer"
        writer = MemmapTensordictWriter(path, NUM_SAMPLES)
        _run_write_read_benchmark(writer)
        shutil.rmtree(path)

@pytest.mark.parametrize("buffer_size", [-1, NUM_SAMPLES //4])
def test_persistent_writer(benchmark, buffer_size, tmp_path: Path):
    @benchmark
    def write_then_read():
        path = tmp_path / "persistent_writer"
        writer = PersistentTensordictWriter(path, NUM_SAMPLES, buffer_size=buffer_size)
        _run_write_read_benchmark(writer)
        shutil.rmtree(path)
