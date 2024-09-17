from __future__ import annotations

from pathlib import Path

import PIL.Image as pil_image
import torch
from unipercept.data import tensors

TEST_ASSETS_PATH = Path(__file__).parent.parent.parent.parent / "assets" / "testing"


def test_pixel_map_registry():
    reg = tensors.registry.pixel_maps
    assert reg is not None

    known_pixel_maps = (
        tensors.OpticalFlow,
        tensors.Mask,
        tensors.Image,
        tensors.PanopticMap,
    )

    assert len(reg) >= len(known_pixel_maps)

    for t in known_pixel_maps:
        assert t in reg, f"{t} not registered in {reg}!"


def test_panoptic_map_to_coco():
    pm = torch.tensor([10, 10, 20], dtype=torch.int).as_subclass(tensors.PanopticMap)
    coco_img, seg_info = pm.to_coco()

    assert len(seg_info) == 2
    assert isinstance(coco_img, pil_image.Image)


def test_panoptic_map_from_file():
    from unipercept.data.sets import catalog

    assert (
        TEST_ASSETS_PATH.exists()
    ), f"Test assets path {TEST_ASSETS_PATH} does not exist!"
    pm_path = TEST_ASSETS_PATH / "truths" / "segmentations" / "0001" / "000000.png"
    assert pm_path.exists(), f"Segmentation file {pm_path} does not exist!"
    pm_info = catalog.get_info("kitti-step")
    assert pm_info is not None, "Catalog info for 'kitti-step' not found!"

    pm = tensors.PanopticMap.read(pm_path, pm_info, format=tensors.LabelsFormat.KITTI)

    assert pm is not None

    sem_ids = pm.get_semantic_map().unique().tolist()
    sem_names = [pm_info.semantic_classes[sem_id].name for sem_id in sem_ids]

    for expected_name in (
        "road",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic sign",
        "vegetation",
        "sky",
        "person",
        "car",
    ):
        assert (
            expected_name in sem_names
        ), f"Expected class {expected_name} not found in {sem_names}!"

    lbls_with_instance = pm.get_instance_map().unique().tolist()
    ins_sem_ids = {
        l // tensors.PanopticMap.DIVISOR for l in lbls_with_instance if l > 0
    }
    ins_trk_ids = {l % tensors.PanopticMap.DIVISOR for l in lbls_with_instance if l > 0}

    assert ins_sem_ids == {13}
    assert ins_trk_ids ^ {1, 2, 3, 4, 5, 6, 7} == set()
