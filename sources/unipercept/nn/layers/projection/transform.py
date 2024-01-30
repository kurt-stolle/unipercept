from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional

import torch
from torch import Tensor, device

from ._se3 import se3_log_map


@torch.jit.script_if_tracing
def _safe_det_3x3(t: Tensor):
    """
    Fast determinant calculation for a batch of 3x3 matrices.

    Note, result of this function might not be the same as `torch.det()`.
    The differences might be in the last significant digit.

    Args:
        t: Tensor of shape (N, 3, 3).

    Returns:
        Tensor of shape (N) with determinants.
    """

    det = (
        t[..., 0, 0] * (t[..., 1, 1] * t[..., 2, 2] - t[..., 1, 2] * t[..., 2, 1])
        - t[..., 0, 1] * (t[..., 1, 0] * t[..., 2, 2] - t[..., 2, 0] * t[..., 1, 2])
        + t[..., 0, 2] * (t[..., 1, 0] * t[..., 2, 1] - t[..., 2, 0] * t[..., 1, 1])
    )

    return det


@torch.jit.script_if_tracing
def _broadcast_bmm(a: Tensor, b: Tensor) -> Tensor:
    """
    Batch multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K) or (M, P, K)
        b: torch tensor of shape (N, K, K)

    Returns:
        a and b broadcast multiplied. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.
    """
    if a.dim() == 2:
        a = a[None]
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = a.expand(len(b), -1, -1)
        if len(b) == 1:
            b = b.expand(len(a), -1, -1)
    return a.bmm(b)


@torch.jit.script_if_tracing
def _axis_angle_rotation(axis: str, angle: Tensor) -> Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


class TransformType(Enum):
    """
    Our implementations depends on TorchScript, for this reason we cannot support
    subclassing of the Transform3D into Rotate, Scale, etc.
    This enumeration is used instead to hardcode the inverse method.
    """

    DEFAULT = 0
    TRANSLATE = 1
    SCALE = 2
    ROTATE = 3


@torch.jit.script_if_tracing
def _inverse(M: Tensor, tt: TransformType) -> Tensor:
    if tt == TransformType.TRANSLATE:
        inv_mask = M.new_ones([1, 4, 4])
        inv_mask[0, 3, :3] = -1.0
        i_matrix = M * inv_mask
        return i_matrix
    elif tt == TransformType.SCALE:
        xyz = torch.stack([M[:, i, i] for i in range(4)], dim=1)
        ixyz = 1.0 / xyz
        imat = torch.diag_embed(ixyz, dim1=1, dim2=2)
        return imat
    elif tt == TransformType.ROTATE:
        return M.permute(0, 2, 1).contiguous()
    else:
        return torch.inverse(M)


@torch.jit.script
class Transform3d:
    def __init__(self, M: Tensor, *, tt: TransformType) -> None:
        """
        Parameters
        ----------
        matrix
            A tensor of shape (4, 4) or of shape (minibatch, 4, 4) representing
            the 4x4 3D transformation matrix.
        """

        if M.ndim not in (2, 3):
            raise ValueError(
                f"'matrix' has to be a 2- or a 3-dimensional tensor, " f"got {M.ndim}!"
            )
        if M.shape[-2] != 4 or M.shape[-1] != 4:
            raise ValueError(
                f"'matrix' has to be a tensor of shape (minibatch, 4, 4) or "
                f"(4, 4), got {M.shape}!"
            )

        self._matrix = M.view(-1, 4, 4)
        self._transforms: List[Any] = []
        self._tt = tt

    def identity_like(self) -> Transform3d:
        """
        Returns a transform with an identity matrix that has the same datatype
        and device as this transform.
        """
        dtype = self._matrix.dtype
        device = self._matrix.device
        return Transform3d(
            torch.eye(4, dtype=dtype, device=device).view(1, 4, 4),
            tt=TransformType.DEFAULT,
        )

    def __len__(self) -> int:
        return self.get_matrix().shape[0]

    def compose(self, others: List[Transform3d]) -> Transform3d:
        """
        Return a new Transform3d representing the composition of self with the
        given other transforms, which will be stored as an internal list.

        Parameters
        ----------
        *others
            Any number of Transform3d objects

        Returns
        -------
            A new Transform3d with the stored transforms
        """

        out = self.clone()
        for t in others:
            assert isinstance(t, Transform3d)
            out._transforms.append(t)

        return out

    def get_matrix(self) -> Tensor:
        """
        Returns a 4×4 matrix corresponding to each transform in the batch.

        If the transform was composed from others, the matrix for the composite
        transform will be returned.
        For example, if self.transforms contains transforms t1, t2, and t3, and
        given a set of points x, the following should be true:

        .. code-block:: python

            y1 = t1.compose(t2, t3).transform(x)
            y2 = t3.transform(t2.transform(t1.transform(x)))
            y1.get_matrix() == y2.get_matrix()

        Where necessary, those transforms are broadcast against each other.

        Returns
        -------
            A (N, 4, 4) batch of transformation matrices representing the
            stored transforms. See the class documentation for the conventions.
        """
        composed_matrix: Tensor = self._matrix.clone()
        if len(self._transforms) > 0:
            for other in self._transforms:
                assert isinstance(other, Transform3d)

                # Recursion is not allowed in TorchScript, so we hardcode
                # a maximum depth.
                # other_matrix = other.get_matrix()
                if len(other._transforms) > 0:
                    raise NotImplementedError("Max. transform depth reached!")

                other_matrix: Tensor = other._matrix.clone()
                # composed_matrix = _broadcast_bmm(composed_matrix, other_matrix)
                composed_matrix = composed_matrix.bmm(other_matrix)
        return composed_matrix

    def get_se3_log(self, eps: float = 1e-4, cos_bound: float = 1e-4) -> Tensor:
        """
        Returns a 6D SE(3) log vector corresponding to each transform in the batch.

        In the SE(3) logarithmic representation SE(3) matrices are
        represented as 6-dimensional vectors `[log_translation | log_rotation]`,
        i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

        The conversion from the 4x4 SE(3) matrix `transform` to the
        6D representation `log_transform = [log_translation | log_rotation]`
        is done as follows:
            ```
            log_transform = log(transform.get_matrix())
            log_translation = log_transform[3, :3]
            log_rotation = inv_hat(log_transform[:3, :3])
            ```
        where `log` is the matrix logarithm
        and `inv_hat` is the inverse of the Hat operator [2].

        See the docstring for `se3.se3_log_map` and [1], Sec 9.4.2. for more
        detailed description.

        Parameters
        ----------
        eps
            A threshold for clipping the squared norm of the rotation logarithm
            to avoid division by zero in the singular case.
        cos_bound
            Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 3 - cos_bound] to avoid non-finite outputs.
            The non-finite outputs can be caused by passing small rotation angles
            to the `acos` function in `so3_rotation_angle` of `so3_log_map`.

        Returns
        -------
            A (N, 6) tensor, rows of which represent the individual transforms
            stored in the object as SE(3) logarithms.

        Raises:
            ValueError if the stored transform is not Euclidean (e.g. R is not a rotation
                matrix or the last column has non-zeros in the first three places).

        [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
        [2] https://en.wikipedia.org/wiki/Hat_operator
        """
        return se3_log_map(self.get_matrix(), eps, cos_bound)

    def _get_matrix_inverse(self) -> torch.Tensor:
        return _inverse(self._matrix, self._tt)

    def inverse(self, invert_composed: bool) -> Transform3d:
        """
        Returns a new Transform3d object that represents an inverse of the
        current transformation.

        Parameters
        ----------
        invert_composed
            - True: First compose the list of stored transformations
                and then apply inverse to the result. This is
                potentially slower for classes of transformations
                with inverses that can be computed efficiently
                (e.g. rotations and translations).
            - False: Invert the individual stored transformations
                independently without composing them.

        Returns
        -------
            A new Transform3d object containing the inverse of the original
            transformation.
        """

        tinv = self.identity_like()

        if invert_composed:
            # first compose then invert
            tinv._matrix = torch.inverse(self.get_matrix())
        else:
            # self._get_matrix_inverse() implements efficient inverse
            # of self._matrix
            i_matrix = self._get_matrix_inverse()

            # 2 cases:
            if len(self._transforms) > 0:
                # a) Either we have a non-empty list of transforms:
                # Here we take self._matrix and append its inverse at the
                # end of the reverted _transforms list. After composing
                # the transformations with get_matrix(), this correctly
                # right-multiplies by the inverse of self._matrix
                # at the end of the composition.
                for t_i in range(len(self._transforms) - 1, -1, -1):
                    t = self._transforms[t_i]
                    assert isinstance(t, Transform3d)

                    # TorchScript does now allow recursion
                    # tinv._transforms.append(t.inverse())
                    assert len(t._transforms) == 0

                    tinv._transforms.append(t._get_matrix_inverse())

                last = self.identity_like()
                last._matrix = i_matrix

                tinv._transforms.append(last)
            else:
                # b) Or there are no stored transformations
                # we just set inverted matrix
                tinv._matrix = i_matrix

        return tinv

    # def stack(self, others: List[Transform3d]) -> Transform3d:
    #     """
    #     Return a new batched Transform3d representing the batch elements from
    #     self and all the given other transforms all batched together.

    #     Args:
    #         *others: Any number of Transform3d objects

    #     Returns:
    #         A new Transform3d.
    #     """
    #     transforms = [self] + others
    #     matrix = torch.cat([t.get_matrix() for t in transforms], dim=0)

    #     return Transform3d(matrix, tt=TransformType.DEFAULT)

    def transform_points(
        self, points: torch.Tensor, eps: Optional[float] = None
    ) -> Tensor:
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before performing the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 3) or (P, 3) depending
            on the dimensions of the transform
        """
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % points.shape)

        N, P, _3 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)

        assert isinstance(self, Transform3d)
        composed_matrix = self.get_matrix()

        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        if points_out.shape[0] == 1 and points.dim() == 2:
            points_out = points_out.reshape(points.shape)

        return points_out

    def transform_normals(self, normals) -> Tensor:
        """
        Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (P, 3) or (N, P, 3) depending
            on the dimensions of the transform
        """
        if normals.dim() not in [2, 3]:
            msg = "Expected normals to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % (normals.shape,))
        composed_matrix = self.get_matrix()

        # TODO: inverse is bad! Solve a linear system instead
        mat = composed_matrix[:, :3, :3]
        normals_out = _broadcast_bmm(normals, torch.inverse(mat.transpose(1, 2)))

        # When transform is (1, 4, 4) and normals is (P, 3) return
        # normals_out of shape (P, 3)
        if normals_out.shape[0] == 1 and normals.dim() == 2:
            normals_out = normals_out.reshape(normals.shape)

        return normals_out

    def clone(self) -> Transform3d:
        """
        Deep copy of Transforms object. All internal tensors are cloned
        individually.

        Returns:
            new Transforms object.
        """
        cloned = Transform3d(self._matrix.clone(), tt=self._tt)
        cloned._matrix = self._matrix.clone()
        for t in self._transforms:
            assert isinstance(t, Transform3d)

            # TorchScript recusion not allowed
            # Unroll manually instead
            # other._transforms.append(t.clone())
            assert len(t._transforms) == 0, "Max. transform depth reached!"

            cloned._transforms.append(Transform3d(t._matrix.clone(), tt=t._tt))

        return cloned


@torch.jit.script_if_tracing
def Translate(
    T,
) -> Transform3d:
    """
    Create a new Transform3d representing 3D translations.

    Option I: Translate(xyz, dtype=torch.float32, device='cpu')
        xyz should be a tensor of shape (N, 3)

    Option II: Translate(x, y, z, dtype=torch.float32, device='cpu')
        Here x, y, and z will be broadcast against each other and
        concatenated to form the translation. Each can be:
            - A python scalar
            - A torch scalar
            - A 1D torch tensor
    """
    N = T.shape[0]

    mat = torch.eye(4, dtype=T.dtype, device=T.device)
    mat = mat.view(1, 4, 4).repeat(N, 1, 1)
    mat[:, 3, :3] = T

    return Transform3d(mat, tt=TransformType.TRANSLATE)


@torch.jit.unused
def Scale(
    x,
    y=None,
    z=None,
    dtype: torch.dtype = torch.float32,
) -> Transform3d:
    """
    A Transform3d representing a scaling operation, with different scale
    factors along each coordinate axis.

    Option I: Scale(s, dtype=torch.float32, device='cpu')
        s can be one of
            - Python scalar or torch scalar: Single uniform scale
            - 1D torch tensor of shape (N,): A batch of uniform scale
            - 2D torch tensor of shape (N, 3): Scale differently along each axis

    Option II: Scale(x, y, z, dtype=torch.float32, device='cpu')
        Each of x, y, and z can be one of
            - python scalar
            - torch scalar
            - 1D torch tensor
    """
    xyz = _handle_input(x, y, z, dtype, device, "scale", allow_singleton=True)
    N = xyz.shape[0]

    # TODO: Can we do this all in one go somehow?
    mat = torch.eye(4, dtype=xyz.dtype, device=xyz.device)
    mat = mat.view(1, 4, 4).repeat(N, 1, 1)
    mat[:, 0, 0] = xyz[:, 0]
    mat[:, 1, 1] = xyz[:, 1]
    mat[:, 2, 2] = xyz[:, 2]
    return Transform3d(mat, tt=TransformType.SCALE)


@torch.jit.script_if_tracing
def Rotate(
    R: Tensor,
) -> Transform3d:
    """
    Create a new Transform3d representing 3D rotation using a rotation
    matrix as the input.

    Args:
        R: a tensor of shape (3, 3) or (N, 3, 3)
        orthogonal_tol: tolerance for the test of the orthogonality of R

    """
    if R.dim() == 2:
        R = R[None]
    if R.shape[-2:] != (3, 3):
        msg = f"R must have shape (3, 3) or (N, 3, 3); got {R.shape}"
        raise ValueError(msg)

    N = R.shape[0]
    mat = torch.eye(4, dtype=R.dtype, device=R.device)
    mat = mat.view(1, 4, 4).repeat(N, 1, 1)
    mat[:, :3, :3] = R
    return Transform3d(mat, tt=TransformType.ROTATE)


@torch.jit.unused
def RotateAxisAngle(
    angle,
    axis: str = "X",
    degrees: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[Device] = None,
) -> Transform3d:
    """
    Create a new Transform3d representing 3D rotation about an axis
    by an angle.

    Assuming a right-hand coordinate system, positive rotation angles result
    in a counter clockwise rotation.

    Args:
        angle:
            - A torch tensor of shape (N,)
            - A python scalar
            - A torch scalar
        axis:
            string: one of ["X", "Y", "Z"] indicating the axis about which
            to rotate.
            NOTE: All batch elements are rotated about the same axis.
    """
    axis = axis.upper()
    if axis not in ["X", "Y", "Z"]:
        msg = "Expected axis to be one of ['X', 'Y', 'Z']; got %s"
        raise ValueError(msg % axis)
    angle = _handle_angle_input(angle, dtype, device, "RotateAxisAngle")
    angle = (angle / 180.0 * torch.pi) if degrees else angle
    # We assume the points on which this transformation will be applied
    # are row vectors. The rotation matrix returned from _axis_angle_rotation
    # is for transforming column vectors. Therefore we transpose this matrix.
    # R will always be of shape (N, 3, 3)
    R = _axis_angle_rotation(axis, angle).transpose(1, 2)

    return Rotate(R)


def _handle_coord(c, dtype: torch.dtype, device: torch.device) -> Tensor:
    """
    Helper function for _handle_input.

    Args:
        c: Python scalar, torch scalar, or 1D torch tensor

    Returns:
        c_vec: 1D torch tensor
    """
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=dtype, device=device)
    if c.dim() == 0:
        c = c.view(1)
    if c.device != device or c.dtype != dtype:
        c = c.to(device=device, dtype=dtype)
    return c


def _handle_input(
    x,
    y,
    z,
    dtype: torch.dtype,
    device: Optional[Device],
    name: str,
    allow_singleton: bool = False,
) -> Tensor:
    """
    Helper function to handle parsing logic for building transforms. The output
    is always a tensor of shape (N, 3), but there are several types of allowed
    input.

    Case I: Single Matrix
        In this case x is a tensor of shape (N, 3), and y and z are None. Here just
        return x.

    Case II: Vectors and Scalars
        In this case each of x, y, and z can be one of the following
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        In this case x, y and z are broadcast to tensors of shape (N, 1)
        and concatenated to a tensor of shape (N, 3)

    Case III: Singleton (only if allow_singleton=True)
        In this case y and z are None, and x can be one of the following:
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        Here x will be duplicated 3 times, and we return a tensor of shape (N, 3)

    Returns:
        xyz: Tensor of shape (N, 3)
    """
    device_ = get_device(x, device)
    # If x is actually a tensor of shape (N, 3) then just return it
    if torch.is_tensor(x) and x.dim() == 2:
        if x.shape[1] != 3:
            msg = "Expected tensor of shape (N, 3); got %r (in %s)"
            raise ValueError(msg % (x.shape, name))
        if y is not None or z is not None:
            msg = "Expected y and z to be None (in %s)" % name
            raise ValueError(msg)
        return x.to(device=device_, dtype=dtype)

    if allow_singleton and y is None and z is None:
        y = x
        z = x

    # Convert all to 1D tensors
    xyz = [_handle_coord(c, dtype, device_) for c in [x, y, z]]

    # Broadcast and concatenate
    sizes = [c.shape[0] for c in xyz]
    N = max(sizes)
    for c in xyz:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r (in %s)" % (sizes, name)
            raise ValueError(msg)
    xyz = [c.expand(N) for c in xyz]
    xyz = torch.stack(xyz, dim=1)
    return xyz


@torch.jit.unused
def _handle_angle_input(
    x, dtype: torch.dtype, device: Optional[Device], name: str
) -> Tensor:
    """
    Helper function for building a rotation function using angles.
    The output is always of shape (N,).

    The input can be one of:
        - Torch tensor of shape (N,)
        - Python scalar
        - Torch scalar
    """
    device_ = get_device(x, device)
    if torch.is_tensor(x) and x.dim() > 1:
        msg = "Expected tensor of shape (N,); got %r (in %s)"
        raise ValueError(msg % (x.shape, name))
    else:
        return _handle_coord(x, dtype, device_)


def _assert_valid_rotation_matrix(R, tol: float = 1e-7) -> None:
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:

    ``RR^T = I and det(R) = 1``
    """

    if R.ndim == 2:
        R = R[None, :, :]

    assert R.ndim == 3
    assert R.shape[-2:] == (3, 3)

    N = R.shape[0]

    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)

    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    det_R = _safe_det_3x3(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))

    assert orthogonal and no_distortion
