from unipercept.vision.geometry import (
    AxesConvention,
    convert_extrinsics,
    rotation_to_axis_angle,
    extrinsics_to_motion,
)
import torch
import pprint


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

    # Check that the conversions are correct (R' == R*)
    M_opencv = extrinsics_to_motion(E_opencv)
    M_opengl = extrinsics_to_motion(E_opengl)
    M_iso8855 = extrinsics_to_motion(E_iso8855)

    for name, M in {
        "(R,t) opencv": M_opencv,
        "(R,t) opengl": M_opengl,
        "(R,t) iso8855": M_iso8855,
    }.items():
        print(f"-- {name} --")
        R, t = M
        a = rotation_to_axis_angle(R)
        pprint.pprint(a.tolist())
        pprint.pprint(t.tolist())

    R_opencv, t_opencv = M_opencv
    #a_opencv = rotation_to_axis_angle(R_opencv)

    R_opengl, t_opengl = M_opengl
    #a_opengl = rotation_to_axis_angle(R_opengl)

    R_iso8855, t_iso8855 = M_iso8855
    #a_iso8855 = rotation_to_axis_angle(R_iso8855)

    #ax, ay, az = a_opencv
    tx, ty, tz = t_opencv
    # OpenCV -> OpenGL : x -> x, y -> -y, z -> -z
    #assert torch.allclose(a_opengl, torch.tensor([ax, -ay, -az]))
    assert torch.allclose(t_opengl, torch.tensor([tx, -ty, -tz]))

    # OpenCV -> ISO8855 : x -> z, y -> -x, z -> -y
    #assert torch.allclose(a_iso8855, torch.tensor([az, -ax, -ay]))
    assert torch.allclose(t_iso8855, torch.tensor([tz, -tx, -ty]))
