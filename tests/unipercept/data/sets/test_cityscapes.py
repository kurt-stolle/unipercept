from __future__ import annotations

import typing
from pathlib import Path

import pytest
from unipercept import file_io, get_dataset
from unipercept.data import collect
from unipercept.utils.dataset import Dataset

SAMPLE_IDS = [
    "aachen_000000_000019",
    "aachen_000001_000019",
    "aachen_000002_000019",
    "aachen_000003_000019",
]


@pytest.fixture(scope="session", params=["cityscapes", "cityscapes-vps"])
def cityscapes_mock(request, tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("cityscapes")
    split = "train"
    ds = get_dataset(request.param)(
        queue_fn=collect.GroupAdjacentTime(1), split=split, root=root.as_posix()
    )

    root_images = Path(ds.path_image)
    root_images.mkdir(parents=True, exist_ok=True)

    root_panoptic = Path(ds.path_panoptic)
    root_panoptic.mkdir(parents=True, exist_ok=True)

    try:
        root_disparity = Path(ds.path_disparity)
    except AttributeError:
        root_disparity = Path(ds.path_depth)

    root_disparity.mkdir(parents=True, exist_ok=True)

    root_camera = root / "camera" / split
    root_camera.mkdir(parents=True, exist_ok=True)

    for sample_id in SAMPLE_IDS:
        image_path = root_images / f"{sample_id}_leftImg8bit.png"
        image_path.touch()

        panoptic_path = root_panoptic / f"{sample_id}_gtFine_panoptic.png"
        panoptic_path.touch()

        disparity_path = root_disparity / f"{sample_id}_disparity.png"
        disparity_path.touch()

        camera_path = root_camera / f"{sample_id}_camera.png"
        camera_path.touch()

    return ds


def test_manifest(cityscapes_mock: Dataset):
    mfst = cityscapes_mock.manifest

    assert "version" in mfst
    assert "timestamp" in mfst
    assert "sequences" in mfst

    assert len(mfst["sequences"]) == len(SAMPLE_IDS)


def test_queue(cityscapes_mock: Dataset):
    queue = cityscapes_mock.queue

    assert len(queue) == len(SAMPLE_IDS)

    for sample_id, items in queue:
        assert isinstance(sample_id, str)
        assert isinstance(items, typing.Mapping)
        for item in items:
            assert item is not None


def test_datapipe(cityscapes_mock: Dataset):
    with pytest.raises(RuntimeError):
        # reads empty files
        item = next(iter(cityscapes_mock.datapipe))
        assert item is None


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_cityscapes_static(split):
    from unipercept.data.sets.cityscapes import CityscapesDataset

    ds_cls = get_dataset("cityscapes")
    assert ds_cls is CityscapesDataset, ds_cls
    if not file_io.isdir(ds_cls.root):
        pytest.skip(
            f"Dataset {ds_cls.__name__} not installed @ {file_io.get_local_path(ds_cls.root)}"
        )
    ds = ds_cls(split=split, queue_fn=collect.GroupAdjacentTime(1))

    assert len(ds.manifest) > 0
    assert len(ds.queue) > 0

    sample_id, sample_item = next(iter(ds.datapipe))
    assert sample_id is not None
    assert sample_item is not None


@pytest.mark.parametrize("all", [True, False])
@pytest.mark.parametrize("pair_size", [1, 2])
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_cityscapes_vps(split, all, pair_size):
    from unipercept.data.sets.cityscapes import CityscapesVPSDataset

    ds_cls = get_dataset("cityscapes-vps")
    assert ds_cls is CityscapesVPSDataset, ds_cls
    if not file_io.isdir(ds_cls.root):
        pytest.skip(
            f"Dataset {ds_cls.__name__} not installed @ {file_io.get_local_path(ds_cls.root)}"
        )
    ds = ds_cls(split=split, queue_fn=collect.GroupAdjacentTime(pair_size), all=all)

    assert len(ds.manifest) > 0
    assert len(ds.queue) > 0

    sample_id, sample_item = next(iter(ds.datapipe))
    assert sample_id is not None
    assert sample_item is not None
