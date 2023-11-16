import pytest
import time
import torch

import unipercept as up
from torch.utils.data import DataLoader


@pytest.mark.parametrize("pin_memory", [True, False])
def benchmark_dataloader(pin_memory):
    # Initialize dataset and dataloader
    dataset = up.data.sets.get_dataset("pascal_voc")(
        split="train", year="2012", queue_fn=up.data.sets.get_default_queue_fn()
    )

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=pin_memory)
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i == 100:
            break
    end_time = time.time()
    print(f"{end_time - start_time:.4f} seconds")
