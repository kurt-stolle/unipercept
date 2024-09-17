from __future__ import annotations

import torch
from torch import Tensor

from unipercept.utils.tensorclass import Tensorclass

from .transform import Rotate, Transform3d, TransformType, Translate

__all__ = ["Cameras", "get_world_to_view_transform"]


class Cameras(Tensorclass):
    """
    Parameters
    ----------
    K
        A calibration matrix of shape (N, 4, 4)
    R
        Rotation matrix of shape (N, 3, 3)
    T
        Translation matrix of shape (N, 3)
    image_size
        A tensor (int32) of shape (N, 2).
    """

    K: Tensor
    R: Tensor
    T: Tensor
    image_size: Tensor

    def get_camera_center(self) -> Tensor:
        w2v_trans = self.get_world_to_view_transform()
        assert isinstance(w2v_trans, Transform3d)

        P = w2v_trans.inverse(True).get_matrix()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)

        C = P[:, 3, :3]
        return C

    def get_world_to_view_transform(self) -> Transform3d:
        world_to_view_transform = get_world_to_view_transform(R=self.R, T=self.T)
        return world_to_view_transform

    def get_full_projection_transform(self) -> Transform3d:
        w2v = self.get_world_to_view_transform()
        v2p = self.get_projection_transform()
        return w2v.compose([v2p])

    def transform_points(self, points, eps: float | None = None) -> Tensor:
        world_to_proj_transform = self.get_full_projection_transform()
        return world_to_proj_transform.transform_points(points, eps=eps)

    def get_projection_transform(self) -> Transform3d:
        transform = Transform3d(
            self.K.transpose(1, 2).contiguous(), tt=TransformType.DEFAULT
        )
        return transform

    def unproject_points(
        self,
        xy_depth: Tensor,
        world_coordinates=True,
    ) -> Tensor:
        if world_coordinates:
            to_camera_transform = self.get_full_projection_transform()
        else:
            to_camera_transform = self.get_projection_transform()

        assert isinstance(to_camera_transform, Transform3d)

        unprojection_transform = to_camera_transform.inverse(True)
        xy_inv_depth = torch.cat((xy_depth[..., :2], 1.0 / xy_depth[..., 2:3]), dim=-1)

        assert isinstance(xy_inv_depth, Tensor)
        assert isinstance(unprojection_transform, Transform3d)

        return unprojection_transform.transform_points(xy_inv_depth)

    def get_principal_point(self) -> Tensor:
        proj_mat = self.get_projection_transform().get_matrix()
        return proj_mat[:, 2, :2]


def get_world_to_view_transform(R: Tensor, T: Tensor) -> Transform3d:
    """
    This function returns a Transform3d representing the transformation
    matrix to go from world space to view space by applying a rotation and
    a translation.

    PyTorch3D uses the same convention as Hartley & Zisserman.
    I.e., for camera extrinsic parameters R (rotation) and T (translation),
    we map a 3D point `X_world` in world coordinates to
    a point `X_cam` in camera coordinates with:
    `X_cam = X_world R + T`

    Parameters
    ----------
    R
        (N, 3, 3) matrix representing the rotation.
    T
        (N, 3) matrix representing the translation.

    Returns
    -------
        Transform3d object which represents the composed RT transformation.
    """
    # TODO: also support the case where RT is specified as one matrix
    # of shape (N, 4, 4).

    if T.shape[0] != R.shape[0]:
        raise ValueError(
            f"Expected R, T to have the same batch dimension; got {R.shape[0]} "
            f", {T.shape[0]}"
        )
    if T.dim() != 2 or T.shape[1:] != (3,):
        raise ValueError(f"Expected T to have shape (N, 3); got {T.shape}")
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        raise ValueError(f"Expected R to have shape (N, 3, 3); got {R.shape}")

    # Create a Transform3d object
    T_ = Translate(T)
    R_ = Rotate(R)

    return R_.compose([T_])
