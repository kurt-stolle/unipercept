import pytest
import torch
from tensordict import TensorDict

from unipercept.engine.writer import PersistentTensordictWriter


def test_writer():
    wr = PersistentTensordictWriter("test.h5", 10)

    for i in range(len(wr)):
        wr.add(TensorDict({"a": torch.randn(1, 3)}, [1]))
