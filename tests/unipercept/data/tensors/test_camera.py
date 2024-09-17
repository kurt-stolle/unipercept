from __future__ import annotations

import pytest
import torch
import torchvision.transforms.v2.functional as tvfn
from unipercept.data.tensors import PinholeCamera

H, W = 256, 512
FX = 10.0
FY = 10.0
CX = W / 2
CY = H / 2


@pytest.fixture()
def pinhole_camera():
    return PinholeCamera.from_parameters([FX, FY], [CX, CY], canvas=[H, W])


def test_project_camera_to_world_consistency(pinhole_camera):
    points_2d = torch.rand(10, 2)
    points_2d[:, 0] *= W
    points_2d[:, 1] *= H

    depths = torch.rand(10, 1) * 100

    points_3d = pinhole_camera.reproject_points(points_2d, depths)
    assert points_3d.shape[0] == points_2d.shape[0]
    assert points_3d.shape[1] == 3

    points_2d_reproj = pinhole_camera.project_points(points_3d)

    for p2d, p3d, p2d_reproj in zip(
        points_2d, points_3d, points_2d_reproj, strict=False
    ):
        print(
            p2d.tolist(), p3d.tolist(), p2d_reproj.tolist(), (p2d - p2d_reproj).tolist()
        )
        assert torch.allclose(p2d, p2d_reproj)


def test_pinhole_camera_resize(pinhole_camera):
    tgt_size = torch.as_tensor([1024, 16]).float()
    cam = tvfn.resize(pinhole_camera, tgt_size.tolist()).as_subclass(PinholeCamera)

    assert torch.allclose(cam.canvas_size, tgt_size), (
        cam.canvas_size.tolist(),
        tgt_size.tolist(),
    )


def test_pinhole_camera_crop(pinhole_camera):
    tgt_off = torch.as_tensor([10, 50]).float()
    tgt_size = torch.as_tensor([H / 2, W / 2]).float()
    cam = tvfn.crop(pinhole_camera, *tgt_off.tolist(), *tgt_size.tolist()).as_subclass(
        PinholeCamera
    )

    print(tgt_size.dtype)
    print(cam.dtype)

    assert torch.allclose(cam.canvas_size, tgt_size), (
        cam.canvas_size.tolist(),
        tgt_size.tolist(),
    )
    tgt_bbox = torch.as_tensor(
        [tgt_off[1], tgt_off[0], tgt_off[1] + tgt_size[1], tgt_off[0] + tgt_size[0]]
    ).float()
    assert torch.allclose(cam.canvas_bbox, tgt_bbox), (
        cam.canvas_bbox.tolist(),
        tgt_bbox.tolist(),
    )
