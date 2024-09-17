from __future__ import annotations

import typing as T

import pytest
import torch.nn as nn
import typing_extensions as TX

import unipercept as up


class SemSegModel(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()
        
        self.backbone = up.nn.backbones.fpn.FPN(
            

        self.conv = nn.Conv2d(in_channels, out_classes, 1)

    def forward(self, x):
        return self.conv(x).softmax(dim=1)


@pytest.fixture(scope="session")
def dataset():
    return up.data.sets.pascal_voc.PascalVOCDataset(
        queue_fn=up.data.collect.ExtractIndividualFrames(), split="train", year="2012"
    )


@pytest.fixture(scope="session")
def model() -> nn.Module:
    return SemSegModel(3, 10)


@pytest.fixture(scope="session")
def model_factory() -> T.Callable[..., nn.Module]:
    def create_model(trial):
        return SemSegModel(3, 10)

    return create_model


@pytest.fixture(scope="session")
def loader_factory(dataset) -> up.data.DataLoaderFactory:
    return up.data.DataLoaderFactory(
        dataset=dataset,
        actions=[],
        sampler=up.data.SamplerFactory("training"),
        config=up.data.DataLoaderConfig(
            drop_last=False,
            # Below are specific settings for testing
            pin_memory=False,
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=None,
        ),
    )
