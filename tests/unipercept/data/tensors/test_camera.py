from unipercept.data.tensors import PinholeCamera
import torchvision.transforms.v2.functional as tvfn

import torch
import pytest

FX = 10.0
FY = 10.0
CX = 64.0
CY = 128.0


@pytest.fixture()
def pinhole_camera():
    return PinholeCamera.from_parameters([FX, FY], [CX, CY], canvas=[256, 512])


def test_pinhole_camera(pinhole_camera):
    mat_i = torch.arange(16).reshape(4, 4).float()
    mat_e = mat_i.clone() + 100
    size_hw = torch.tensor([512, 256]).float()

    cam = PinholeCamera.from_parts(mat_i, mat_e, size_hw)

    assert pinhole_camera.shape[-1] == 10
    assert pinhole_camera.shape[-2] == 4
    assert (cam.intrinsic_matrix == mat_i).all(), (cam.intrinsic_matrix, mat_i)
    assert (cam.extrinsic_matrix == mat_e).all(), (cam.extrinsic_matrix, mat_e)
    assert torch.allclose(cam.canvas_size, size_hw.float()), (cam.canvas_size, size_hw)


def test_pinhole_camera_resize(pinhole_camera):
    tgt_size = torch.as_tensor([1024, 16]).float()
    cam = tvfn.resize(pinhole_camera, tgt_size.tolist()).as_subclass(PinholeCamera)

    print(cam.camera_matrix.tolist())
    print(cam.canvas_bbox.tolist())
    print(cam.canvas_size.tolist())

    assert torch.allclose(cam.canvas_size, tgt_size), (
        cam.canvas_size.tolist(),
        tgt_size.tolist(),
    )


def test_pinhole_camera_crop(pinhole_camera):
    tgt_off = torch.as_tensor([10, 50]).float()
    tgt_size = torch.as_tensor([256, 128]).float()
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
