from __future__ import annotations

import typing as T

import pytest
import torch
import torchvision.transforms.v2 as transforms
import typing_extensions as TX

from unipercept.data import ops
from unipercept.data.tensors import registry
from unipercept.model import CameraModel, CaptureData, InputData, MotionData

FAKE_PARAMS = [(1, 1), (1, 2), (3, 1), (3, 2)]


@pytest.fixture(params=FAKE_PARAMS, ids=[f"batch_{b}-pair_{p}" for b, p in FAKE_PARAMS])
def fake_data(request) -> tuple[list[str], InputData]:
    batch_size, pair_size = request.param
    ids = [f"fake_{i}" for i in range(batch_size)]
    datas = [
        InputData(
            cameras=CameraModel(
                image_size=torch.as_tensor((1024, 1024)),
                matrix=torch.randn(pair_size, 4, 4),  # type: ignore
                pose=torch.randn(pair_size, 4, 4),  # type: ignore
                batch_size=[],
            ),
            motions=None,
            captures=CaptureData(
                times=torch.linspace(1, 10, pair_size),
                images=torch.randn(pair_size, 3, 1024, 1024).as_subclass(Image),  # type: ignore
                segmentations=torch.randn(pair_size, 1024, 1024).as_subclass(PanopticMap),  # type: ignore
                depths=torch.randn(pair_size, 1024, 1024).as_subclass(DepthMap),  # type: ignore
                batch_size=[pair_size],
            ),
            batch_size=[],
        )
        for _ in range(batch_size)
    ]

    data: InputData = torch.stack(datas)  # type: ignore

    assert data.batch_size == torch.Size((batch_size,)), data.batch_size or "N/A"
    assert data.captures.batch_size == torch.Size((batch_size, pair_size)), (
        data.captures.batch_size or "N/A"
    )
    assert data.captures.images.shape == torch.Size(
        (batch_size, pair_size, 3, 1024, 1024)
    ), (data.images.shape or "N/A")

    return ids, data


def test_op_nop(fake_data):
    op = _D.ops.NoOp()
    ids, x = fake_data
    ids, y = op(ids, x)

    assert x is y


def test_torchvision_random_resized_crop(fake_data):
    from torch.utils._pytree import tree_flatten

    tr = transforms.RandomResizedCrop(512, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    op = ops.TorchvisionOp(tr)

    ids, x = fake_data
    x = x.fix_subtypes_()

    print(f"Input: {type(x)} w/ captures {type(x.captures)}: {x}")
    ids, y = op(ids, x)

    print(f"Output: {type(y)} w/ captures {type(y.captures)}: {y}")

    assert y.batch_size == x.batch_size
    assert y.captures.batch_size == y.captures.batch_size

    for k, y_cap in y.captures.fix_subtypes_().items():
        if type(y_cap) in registry.pixel_maps:
            assert y_cap.shape[-2:] == (
                512,
                512,
            ), f"Shape mismatch for {k}: {y_cap.shape}"


def test_torchvision_random_resized_crop(fake_data):
    from torch.utils._pytree import tree_flatten

    tr = transforms.RandomResizedCrop(512, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    op = ops.TorchvisionOp(tr)

    ids, x = fake_data
    x = x.fix_subtypes_()

    print(f"Input: {type(x)} w/ captures {type(x.captures)}: {x}")
    ids, y = op(ids, x)

    print(f"Output: {type(y)} w/ captures {type(y.captures)}: {y}")

    assert y.batch_size == x.batch_size
    assert y.captures.batch_size == y.captures.batch_size

    for k, y_cap in y.captures.fix_subtypes_().items():
        if type(y_cap) in registry.pixel_maps:
            assert y_cap.shape[-2:] == (
                512,
                512,
            ), f"Shape mismatch for {k}: {y_cap.shape}"
