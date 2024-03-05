from __future__ import annotations

import typing as T

import PIL.Image as pil_image
import torch
import typing_extensions as TX

from unipercept.data import tensors


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
