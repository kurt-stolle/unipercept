from __future__ import annotations

import torch
from unipercept.vision.geometry import (
    AxesConvention,
    convert_extrinsics,
    convert_points,
)


def test_convert_extrinsics():
    E_opencv = torch.eye(4)
    E_opencv[:3, :3] = torch.randn(3, 3)
    E_opencv[:3, 3] = torch.randn(3)

    # OpenCV -> OpenGL -> OpenCV
    E_opengl = convert_extrinsics(E_opencv, tgt=AxesConvention.OPENGL)
    E_opengl_opencv = convert_extrinsics(E_opengl, src=AxesConvention.OPENGL)

    # OpenCV -> ISO8855 -> OpenCV
    E_iso8855 = convert_extrinsics(E_opencv, tgt=AxesConvention.ISO8855)
    E_iso8855_opencv = convert_extrinsics(E_iso8855, src=AxesConvention.ISO8855)

    # ISO8855 -> OpenGL
    E_iso8855_opengl = convert_extrinsics(
        E_iso8855,
        src=AxesConvention.ISO8855,
        tgt=AxesConvention.OPENGL,
    )

    # Check that the conversions are consistent (R -> R' -> R)
    assert not torch.allclose(E_opencv, E_opengl)
    assert torch.allclose(E_opencv, E_opengl_opencv)
    assert not torch.allclose(E_opencv, E_iso8855)
    assert torch.allclose(E_opencv, E_iso8855_opencv)
    assert torch.allclose(E_opengl, E_iso8855_opengl)


def test_convert_points():
    right = 1.0
    left = -right
    down = 2.0
    up = -down
    fwd = 3.0
    bwd = -fwd

    P_cv = torch.tensor([[right, down, fwd]], dtype=torch.float32)
    P_cv_gl = convert_points(P_cv, tgt=AxesConvention.OPENGL)
    P_cv_gl_cv = convert_points(P_cv_gl, src=AxesConvention.OPENGL)

    assert not torch.allclose(P_cv, P_cv_gl)
    assert torch.allclose(P_cv, P_cv_gl_cv)

    P_cv_iso = convert_points(P_cv, tgt=AxesConvention.ISO8855)
    P_cv_iso_cv = convert_points(P_cv_iso, src=AxesConvention.ISO8855)

    assert not torch.allclose(P_cv, P_cv_iso)
    assert torch.allclose(P_cv, P_cv_iso_cv)

    P_cv_iso_gl = convert_points(
        P_cv_iso, src=AxesConvention.ISO8855, tgt=AxesConvention.OPENGL
    )
    assert torch.allclose(P_cv_gl, P_cv_iso_gl)

    xyz_cv = tuple(P_cv[0].tolist())
    xyz_gl = tuple(P_cv_gl[0].tolist())
    xyz_iso = tuple(P_cv_iso[0].tolist())

    assert xyz_cv == (right, down, fwd)
    assert xyz_gl == (right, up, bwd)
    assert xyz_iso == (fwd, left, up)
