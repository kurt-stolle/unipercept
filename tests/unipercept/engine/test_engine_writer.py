from __future__ import annotations

import shutil
from pathlib import Path

import torch
from tensordict import TensorDict
from unipercept.engine.writer import ResultsWriter

NUM_SAMPLES = 32


def _run_write_read_benchmark(writer: ResultsWriter):
    for _ in range(NUM_SAMPLES):
        writer.add(
            TensorDict(
                {"a": torch.randn(1, 64, 120), "z": torch.ones(1, 1)}, batch_size=[1]
            )
        )
    writer.flush()

    td = writer.tensordict

    for i in range(NUM_SAMPLES):
        item = td.get_at("a", i)
        assert item.shape == (64, 120)

    writer.close()


def test_memmap_writer(benchmark, tmp_path: Path):
    @benchmark
    def write_then_read():
        path = tmp_path / "memmap_writer"
        writer = ResultsWriter(path, NUM_SAMPLES, write_offset=0)
        _run_write_read_benchmark(writer)
        shutil.rmtree(path)
