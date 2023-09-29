from pathlib import Path
from pprint import pprint
from typing import Any

import pytest
import torch
from detectron2.config import instantiate
from torch.profiler import ProfilerActivity, profile, record_function


@pytest.mark.integration
def test_config_train(cfg, device):
    # Initialize model
    cfg_model = cfg.model
    model = instantiate(cfg_model)
    assert model is not None
    assert isinstance(model, torch.nn.Module), type(model)

    model = model.train().to(device=device)

    # Edit configuration to make dataloader synchronous
    cfg_dataloader = cfg.dataloader
    cfg_dataloader.train.num_workers = 0
    cfg_dataloader.train.total_batch_size = 2

    # Initialize dataloader
    try:
        dl = instantiate(cfg_dataloader.test)
    except KeyError:
        pytest.skip("Dataset is not registered on the current machine.")
    assert dl is not None

    # Sample a test input
    ins: dict[str, Any]
    ins = next(iter(dl))
    assert ins is not None

    # Run model forward pass
    res: dict[str, Any]
    res = model(ins)

    assert res is not None
    assert isinstance(res, dict), type(res)

    # Run model backward pass
    loss = torch.stack(list(res.values())).sum()
    loss.backward()
    loss.detach_()

    with torch.no_grad():
        grads = torch.stack([p.grad.sum() for p in model.parameters() if p.grad is not None])
        grads.detach_()

    # Print output stats
    stats = {
        "Losses": {k: v.detach().cpu().item() for k, v in res.items()},
        "Loss sum": loss.detach().cpu().item(),
        "Gradient sum": grads.sum().cpu().item(),
        "Gradient abs mean": grads.abs().mean().cpu().item(),
        "Grad abs norm": grads.abs().norm().cpu().item(),
    }

    print("-" * 5, "Outputs")
    for name, values in stats.items():
        if isinstance(values, dict):
            print(name)
            for k, v in values.items():
                print(f"- {k:28s}: {v:.4f}")
        else:
            print(f"{name:30s}: {values}")


@pytest.mark.integration
def test_config_eval(cfg, device):
    # Initialize model
    cfg_model = cfg.model
    model = instantiate(cfg_model)
    assert model is not None
    assert isinstance(model, torch.nn.Module), type(model)

    model = model.eval().to(device=device)

    # Edit configuration to make dataloader synchronous
    cfg_dataloader = cfg.dataloader
    cfg_dataloader.test.batch_size = 2
    cfg_dataloader.test.num_workers = 0

    # Initialize dataloader
    try:
        dl = instantiate(cfg_dataloader.test)
    except KeyError:
        pytest.skip("Dataset is not registered on the current machine.")
    assert dl is not None

    # Sample a test input
    ins: dict[str, Any]
    ins = next(iter(dl))
    assert ins is not None

    # Run model forward pass
    res: dict[str, Any]
    res = model(ins)

    assert res is not None
    assert isinstance(res, list), type(res)

    for res_item in res:
        assert isinstance(res_item, dict), type(res_item)
        assert res_item.keys() == model.output_keys, f"Expected outputs {model.output_keys}, got {res_item.keys()}"
