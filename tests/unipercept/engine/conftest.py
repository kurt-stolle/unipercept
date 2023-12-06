from __future__ import annotations

import typing as T

import pytest
import torch.nn as nn
from unimodels import toys

import unipercept as up


@pytest.fixture(scope="session")
def dataset():
    return up.data.sets.pascal_voc.PascalVOCDataset(
        queue_fn=up.data.collect.ExtractIndividualFrames(), split="train", year="2012"
    )


@pytest.fixture(scope="session")
def model() -> nn.Module:
    return toys.SemanticSegmenter(10, 3)


@pytest.fixture(scope="session")
def model_factory() -> T.Callable[..., nn.Module]:
    def create_model(trial):
        return toys.SemanticSegmenter(10, 3)

    return create_model


@pytest.fixture(scope="session")
def loader_factory(dataset) -> up.data.DataLoaderFactory:
    config = up.data.DataLoaderConfig(batch_size=2, drop_last=False, pin_memory=True, num_workers=4)
    factory = up.data.DataLoaderFactory(
        dataset=dataset, actions=[], sampler=up.data.SamplerFactory("training"), config=config
    )

    return factory
