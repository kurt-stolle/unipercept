import pytest
import torch
from detectron2.config import instantiate
from omegaconf import DictConfig


def test_instantiate_model(cfg):
    ins = instantiate(cfg.model)
    assert ins is not None
    assert isinstance(ins, torch.nn.Module), type(ins)


def test_instantiate_trainer(cfg):
    cfg = cfg.train

    assert cfg is not None
    for attr in ["device", "output_dir", "max_iter", "ddp", "amp", "init_checkpoint", "checkpointer"]:
        assert hasattr(cfg, attr), attr


@pytest.mark.parametrize("loader_key", ["train", "test"])
def test_instantiate_loader(cfg, loader_key):
    import torch.utils.data
    from unipercept.data import DataLoaderFactory
    from unipercept.data.points import InputData

    lf = instantiate(cfg.data.loaders[loader_key])
    assert lf is not None
    assert isinstance(lf, DataLoaderFactory), type(lf)

    ld = lf(1)

    id, item = next(iter(ld))

    assert id is not None, type(id)
    assert isinstance(item, InputData), type(item)


def test_instantiate_evaluators(cfg):
    evaluator = instantiate(cfg.data.evaluator)
    assert evaluator is not None
    print(evaluator)
