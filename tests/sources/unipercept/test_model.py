"""
Tests for the `unipercept.model` module.
"""

import torch

from unipercept.model import ModelOutput


def test_model_output():
    def mock_output(i) -> ModelOutput:
        return ModelOutput(losses={"loss": 1.0}, metrics={"acc": i}, predictions={}, batch_size=[])

    out_prealloc = ModelOutput(batch_size=[3])

    for i in range(3):
        out_prealloc[i] = mock_output(i)

    assert out_prealloc.metrics["acc"].shape == torch.Size((3,)), out_prealloc.metrics["acc"].shape
    assert out_prealloc.losses["loss"].shape == torch.Size((3,)), out_prealloc.losses["loss"].shape

    assert out_prealloc.metrics["acc"].tolist() == [0, 1, 2], out_prealloc.metrics["acc"].tolist()
    assert out_prealloc.losses["loss"].tolist() == [1, 1, 1], out_prealloc.losses["loss"].tolist()
    assert out_prealloc.metrics["acc"].sum() == 1 + 2 + 0, out_prealloc.metrics["acc"].sum()
    assert out_prealloc.losses["loss"].sum() == 1 + 1 + 1, out_prealloc.losses["loss"].sum()
