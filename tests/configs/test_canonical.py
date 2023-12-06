import pytest

import unipercept as up


def test_instantiate_model(cfg):
    up.create_model(cfg)


def test_instantiate_engine(cfg):
    up.create_engine(cfg)


def test_instantiate_data(cfg):
    up.create_loaders(cfg)

    from unipercept.data import DataLoaderFactory
    from unipercept.model import InputData

    for loader_key in cfg.data.loaders:
        lf = up.config.instantiate(cfg.data.loaders[loader_key])
        assert lf is not None
        assert isinstance(lf, DataLoaderFactory), type(lf)

        try:
            ld = lf(1)
            id, item = next(iter(ld))
        except FileNotFoundError:
            pytest.xfail("Dataset is not installed: {lf}")

        assert id is not None, type(id)
        assert isinstance(item, InputData), type(item)
