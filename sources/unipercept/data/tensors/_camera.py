r"""
Implements the camera s a TVTensor.
"""

from __future__ import annotations
import pprint
import typing as T
import typing_extensions as TX
import torch
from torch import Tensor, Size

from torchvision.transforms.v2.functional import register_kernel
from torchvision.tv_tensors import TVTensor

from torchvision.transforms.v2.functional._geometry import _compute_resized_output_size

__all__ = ["PinholeCamera"]


class PinholeCamera(TVTensor):
    """
    A PinholeCamera is a TVTensor that represents a pinhole camera model.
    This tensor always has a shape of [..., 4, 9], where the last dimension
    represents a concatenation of the intrinsic and extrinsic parameters of the camera,
    plus an additional column that represents the image size:

    - [..., :, 0:4] represents the intrinsic matrix
    - [..., :, 4:8] represents the extrinsic matrix
    - [..., :, 8:9] represents the canvas as (u1, v1, u2, v2)
    """

    ###################
    # Factory methods #
    ###################

    @classmethod
    def with_defaults_as(
        cls,
        like: Tensor,
        n_cameras: int = 1,
    ) -> T.Self:
        """
        Constructs a PinholeCamera intiialized with heuristic values based on the input
        tensor, which we assume has the same spatial dimensions as the image.
        This should usually be avoided when the user intends to use the camera for
        rendering purposes.
        """
        if like.ndim >= 5:
            msg = (
                "Default initialization is not supported for batched tensors. This is "
                "due to that there could potentially be padding and cropping operations "
                f"that would affect the camera parameters. Got image shape: {like.shape}"
            )
            raise NotImplementedError(msg)

        H, W = like.shape[-2:]
        F = torch.tensor([1.0, 1.0])
        P = torch.tensor([W / 2, H / 2])

        cam = cls.from_parameters(F, P, canvas=[0, 0, W, H])
        cam = cam.unsqueeze(0)
        cam = cam.expand((n_cameras, -1, -1))
        return cam.as_subclass(cls)

    @classmethod
    def from_parameters(
        cls,
        focal_length: Tensor | T.Iterable[int | float],
        principal_point: Tensor | T.Iterable[float | int],
        rotation: Tensor | T.Iterable[float | int] | None = None,  # pitch, yaw, roll
        translation: Tensor | T.Iterable[float | int] | None = None,  # x, y, z
        canvas: Tensor | T.Iterable[float | int] | None = None,
    ) -> T.Self:
        """
        Constructs a PinholeCamera tensor from the camera parameters.
        """
        K = cls.build_intrinsic_matrix(focal_length, principal_point, orthographic=False)
        if rotation is None and translation is None:
            E = torch.zeros_like(K)
        else:
            if rotation is None:
                translation = torch.as_tensor(translation)
                rotation = torch.zeros_like(translation)
            else:
                rotation = torch.as_tensor(rotation)
                translation = torch.as_tensor(translation)
            E = cls.build_extrinsic_matrix(rotation, translation)

        return cls.from_parts(K, E, canvas)

    @classmethod
    def from_parts(
        cls,
        intrinsics: Tensor,
        extrinsics: Tensor | None = None,
        canvas: Tensor | None = None,
    ) -> T.Self:
        """
        Constructs a PinholeCamera tensor from the intrinsic and extrinsic matrices
        and the image size.
        """
        intrinsics = torch.as_tensor(intrinsics)
        assert intrinsics.ndim >= 2 and intrinsics.size(-1) == 4

        if extrinsics is None:
            extrinsics = torch.zeros_like(intrinsics)
        else:
            extrinsics = torch.as_tensor(extrinsics)
        assert extrinsics.ndim >= 2 and extrinsics.size(-1) == 4

        if canvas is None:
            u0, v0 = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
            canvas = torch.cat([v0 * 2, u0 * 2], dim=-1)
        else:
            canvas = torch.as_tensor(canvas).to(intrinsics)
            if canvas.size(-1) == 2:
                canvas = canvas.flip(-1)  # H, W -> W, H
                canvas = torch.cat([torch.zeros_like(canvas), canvas], dim=-1)  # bbox
            elif canvas.size(-1) != 4:
                msg = (
                    "Canvas must be a tensor of shape (V, U) or (u1, v1, u2, v2). "
                    f"Got: {canvas.shape}"
                )
                raise ValueError(msg)

        assert canvas.ndim == intrinsics.ndim - 1

        res = torch.zeros((*intrinsics.shape[:-2], 4, 10), dtype=intrinsics.dtype)
        res[..., :, 0:4] = intrinsics
        res[..., :, 4:8] = extrinsics
        res[..., :, 8] = canvas
        return res.as_subclass(cls)
    
    
    @staticmethod
    def build_extrinsic_matrix(
        rotation: T.List[T.Tuple[float, float, float]] | Tensor,
        translation: T.List[T.Tuple[float, float, float]] | Tensor,
    ) -> Tensor:
        r"""
        Build the extrinsic matrix (R|t) using the rotations and translations.
        """

        def _pyr_to_rot(pyr: Tensor) -> Tensor:
            """
            Convert Pitch Yaw Roll values to a 3x3 rotation matrix.
            """
            pitch, yaw, roll = pyr.unbind(-1)
            sin_p = torch.sin(pitch)
            cos_p = torch.cos(pitch)
            sin_y = torch.sin(yaw)
            cos_y = torch.cos(yaw)
            sin_r = torch.sin(roll)
            cos_r = torch.cos(roll)

            R_x = torch.tensor(
                [
                    [1, 0, 0],
                    [0, cos_p, -sin_p],
                    [0, sin_p, cos_p],
                ],
                device=pyr.device,
            )
            R_y = torch.tensor(
                [
                    [cos_y, 0, sin_y],
                    [0, 1, 0],
                    [-sin_y, 0, cos_y],
                ],
                device=pyr.device,
            )
            R_z = torch.tensor(
                [
                    [cos_r, -sin_r, 0],
                    [sin_r, cos_r, 0],
                    [0, 0, 1],
                ],
                device=pyr.device,
            )
            return R_z @ R_y @ R_x

        rotation = torch.as_tensor(rotation).float()
        translation = torch.as_tensor(translation).float()

        # Amount of cameras
        ndim = rotation.ndim
        assert ndim == translation.ndim
        if ndim == 1:
            rotation = rotation.unsqueeze(0)
            translation = translation.unsqueeze(0)

        N = rotation.size(0)

        # Value checks
        assert rotation.shape[-1] == 3, rotation.shape
        assert translation.shape[-1]== 3, translation.shape

        # Create extrinsic matrix [R|t] + [0 0 0 1]
        extrinsic_matrix = torch.zeros(N, 4, 4, device=rotation[0].device)
        for i in range(N):
            extrinsic_matrix[i, :3, :3] = _pyr_to_rot(rotation[i])
            extrinsic_matrix[i, :3, 3] = translation[i]
            extrinsic_matrix[i, 3, 3] = 1.0

        if ndim == 1:
            extrinsic_matrix = extrinsic_matrix.squeeze(0)

        return extrinsic_matrix

    @staticmethod
    def build_intrinsic_matrix(
        focal_length: T.List[T.Tuple[float, float]] | Tensor,
        principal_point: T.List[T.Tuple[float, float]] | Tensor,
        orthographic: bool,
    ) -> Tensor:
        r"""
        Build the calibration matrix (K-matrix) using the focal lengths and
        principal points.
        """

        # Focal length, shape (), (N, 1) or (N, 2)
        if not isinstance(focal_length, Tensor):
            focal_length = torch.as_tensor(focal_length).float()
        if focal_length.shape[-1] == 1:
            fx = fy = focal_length
        else:
            fx, fy = focal_length.unbind(-1)

        # Principal point
        if not isinstance(principal_point, Tensor):
            principal_point = torch.as_tensor(principal_point).float()
        px, py = principal_point.unbind(-1)

        # Batched/unbatched
        ndim = focal_length.ndim
        assert ndim == principal_point.ndim
        if ndim == 1:
            fx = fx.unsqueeze(0)
            fy = fy.unsqueeze(0)
            px = px.unsqueeze(0)
            py = py.unsqueeze(0)

        # Amount of cameras
        N = fx.size(0)

        # Create calibration matrix
        K: Tensor = fx.new_zeros(N, 4, 4)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        if orthographic:
            K[:, 0, 3] = px
            K[:, 1, 3] = py
            K[:, 2, 2] = 1.0
            K[:, 3, 3] = 1.0
        else:
            K[:, 0, 2] = px
            K[:, 1, 2] = py
            K[:, 3, 2] = 1.0
            K[:, 2, 3] = 1.0

        if ndim == 1:
            K = K.squeeze(0)

        return K


    def __new__(
        cls,
        data: T.Any,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> T.Self:
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        assert cls.check_pinhole_camera_tensor_shape(
            tensor
        ), f"Expected a tensor of shape [..., 2, 4, 4], but got {tensor.shape}"
        return tensor.as_subclass(cls)

    @staticmethod
    def check_pinhole_camera_tensor_shape(tensor: Tensor) -> bool:
        """
        Returns whether the input tensor has the shape of a PinholeCamera tensor.

        This is not a sufficient condition for assessing whether the tensor is a
        PinholeCamera. It only checks the shape of the tensor and whether this would
        be a valid shape for a PinholeCamera tensor.
        """
        return tensor.size(-2) == 4 and tensor.size(-1) == 9

    ########################
    # Tensor destructuring #
    ########################

    @property
    def intrinsic_matrix(self) -> Tensor:
        """
        Returns the intrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return self[..., :, 0:4]

    @property
    def extrinsic_matrix(self) -> Tensor:
        """
        Returns the extrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return self[..., :, 4:8]

    @property
    def canvas_size(self) -> Tensor:
        """
        Returns the image size as a (batch_size, (V, U)) tensor.
        """
        bbox = self.canvas_bbox
        size = bbox[..., 2:] - bbox[..., :2]
        return size.flip(-1)

    @property
    def canvas_height(self) -> Tensor:
        """
        Returns the image height as a (batch_size, V) tensor.
        """
        return self.canvas_bbox[..., 3] - self.canvas_bbox[..., 1]

    @property
    def canvas_width(self) -> Tensor:
        """
        Returns the image width as a (batch_size, U) tensor.
        """
        return self.canvas_bbox[..., 2] - self.canvas_bbox[..., 0]

    @property
    def canvas_center(self) -> Tensor:
        """
        Returns the image center as a (batch_size, (V, U)) tensor.
        """
        bbox = self.canvas_bbox
        size = self.canvas_size
        return bbox[..., :2].flip(-1) + size / 2

    @property
    def canvas_bbox(self) -> Tensor:
        """
        Returns the image crop bounding box as a [B, (u1, v1, u2, v2)] tensor.
        """
        return self[..., 8]

    ##########################
    # Sub-matrices and views #
    ##########################

    @property
    def focal_length(self) -> Tensor:
        """
        Returns the focal length of the camera as a [..., (fx, fy)] tensor.
        """
        return self[..., :2, :2].diagonal(0,-2,-1)

    @property
    def principal_point(self) -> Tensor:
        """
        Returns the principal point of the camera as a [..., (u0, v0)] tensor.
        """
        return self[..., :2, 2]

    @property
    def camera_matrix(self) -> Tensor:
        """
        Returns the camera matrix of the camera.
        """
        return self[..., :3, 0:3]

    @property
    def rotation_matrix(self) -> Tensor:
        """
        Returns the rotation part of the extrinsic parameters as a (batch_size, 3, 3) tensor.
        """
        return self[..., :3, 4:7]

    @property
    def translation_vector(self) -> Tensor:
        """
        Returns the translation part of the extrinsic parameters as a (batch_size, 3) tensor.
        """
        return self[..., :3, 4:7]

    @property
    def rotation_translation_matrix(self) -> Tensor:
        """
        Returns the concatenated rotation-translation matrix of the camera as a
        (batch_size, 3, 4) tensor.
        """
        return self[..., :3, 4:8]
    
    ##########################
    # Representation methods #
    ##########################
    
    def to_map(self, like: Size | T.Iterable[int] | Tensor) -> Tensor:
        r"""
        Encode the camera intrinsic parameters (focal length and principle point) to a map
        with channels for the distance to the principle point and the field of view.
        """
        if isinstance(like, Tensor):
            *_, height, width = like.shape
        else:
            *_, height, width = like
        fx, fy = self.focal_length.unbind(-1)
        u0, v0 = self.principal_point.unbind(-1)
        f = (fx + fy) / 2.0
        # principle point location
        x_row = torch.arange(0, width, dtype=torch.float32)
        x_row_center_norm = (x_row - u0) / width
        x_center = torch.tile(x_row_center_norm, (height, 1))  # [H, W]

        y_col = torch.arange(0, height, dtype=torch.float32)
        y_col_center_norm = (y_col - v0) / height
        y_center = torch.tile(y_col_center_norm, (width, 1)).T

        # FoV
        fov_x = torch.arctan(x_center / (f / width))
        fov_y = torch.arctan(y_center / (f / height))

        return torch.stack([x_center, y_center, fov_x, fov_y], dim=2)


    #####################
    # Modifying methods #
    #####################

    def scale_(self, factor: Tensor) -> T.Self:
        """
        This is the inplace variant of :meth:`scale`.
        """
        if factor.size(-1) == 1:
            factor = factor.expand(*factor.shape[:-1], 2)
        elif factor.size(-1) != 2:
            msg = f"Expected a tensor of shape [..., (V, U)], but got {factor.shape}"
            raise ValueError(msg)

        factor = factor.to(self)
        scale_h, scale_w = factor[..., 0], factor[..., 1]

        # Modify the intrinsic matrix
        self.intrinsic_matrix[..., 0, 0] *= scale_w
        self.intrinsic_matrix[..., 1, 1] *= scale_h
        self.intrinsic_matrix[..., 0, 2] *= scale_w
        self.intrinsic_matrix[..., 1, 2] *= scale_h

        # Modify the crop (and convert H, W -> V, U)
        crop_center = (self.canvas_center * factor).flip(-1)
        crop_size = (self.canvas_size * factor).flip(-1)

        self[..., 8] = torch.cat(
            [crop_center - crop_size / 2, crop_center + crop_size / 2], dim=-1
        )

        return self

    def scale(self, factor: Tensor, inplace: bool = False) -> T.Self:
        """
        Scales the camera by the given factor.

        Assumes the scaling is done w.r.t. the center of the current image crop.
        """
        if not inplace:
            self = self.clone().as_subclass(type(self))
        return self.scale_(factor)

    def crop_(self, crop: Tensor) -> T.Self:
        """
        This is the inplace variant of :meth:`crop`.
        """
        self[..., 8] = crop.to(self)
        return self

    def crop(self, crop: Tensor, inplace: bool = False) -> T.Self:
        """
        Applies a crop to the camera.

        Parameters
        ----------
        crop: Tensor[..., (u1, v1, u2, v2)]
            The crop to apply to the camera.
        """
        if not inplace:
            self = self.clone().as_subclass(type(self))
        return self.crop_(crop)

    @TX.override
    def __repr__(self):
        name = self.__class__.__name__
        if name == "Tensor":
            name = "PinholeCamera"
        kwds = ", ".join(
            f"{k}={v}"
            for k, v in {
                "shape": self.shape,
                "device": self.device,
                "requires_grad": self.requires_grad,
            }.items()
        )
        self = self.detach().cpu().as_subclass(PinholeCamera)
        TAB = " " * 2
        fields = f"\n{TAB}".join(
            f"{k}: {v}"
            for k, v in {
                "focal_length": self.focal_length.tolist(),
                "principal_point": self.principal_point.tolist(),
                "rotation": self.rotation_matrix.tolist(),
                "translation": self.translation_vector.tolist(),
                "canvas_size": self.canvas_size.tolist(),
                "canvas_center": self.canvas_center.tolist(),
            }.items()
        )
        return f"{self.__class__.__name__}({self.dtype}, {kwds}){{\n{TAB}{fields}\n}}"

@register_kernel(functional="resize", tv_tensor_cls=PinholeCamera)
def resize_camera(
    camera: PinholeCamera,
    size: T.List[int],
    interpolation: T.Any = None,  # noqa: U100
    max_size: int | None = None,
    antialias: T.Any = False,  # noqa: U100
    use_rescale: bool = False,
) -> Tensor:
    h_old, w_old = camera.canvas_size.tolist()
    h_new, w_new = _compute_resized_output_size(
        (h_old, w_old), size=size, max_size=max_size
    )

    return camera.scale(torch.tensor([h_new / h_old, w_new / w_old]))


@register_kernel(functional="crop", tv_tensor_cls=PinholeCamera)
def crop_camera(
    camera: PinholeCamera,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tensor:
    crop = torch.tensor([left, top, left + width, top + height])
    return camera.crop(crop)

def generate_rays(
    K: Tensor, image_shape: T.Tuple[int, int], noisy: bool = False
):
    batch_size, device, dtype = (
        K.shape[0],
        K.device,
        K.dtype,
    )
    height, width = image_shape
    # Generate grid of pixel coordinates
    pixel_coords_x = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    pixel_coords_y = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if noisy:
        pixel_coords_x += torch.rand_like(pixel_coords_x) - 0.5
        pixel_coords_y += torch.rand_like(pixel_coords_y) - 0.5
    pixel_coords = torch.stack(
        [pixel_coords_x.repeat(height, 1), pixel_coords_y.repeat(width, 1).t()], dim=2
    )  # (H, W, 2)
    pixel_coords = pixel_coords + 0.5

    # Calculate ray directions
    intrinsics_inv = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics_inv[:, 0, 0] = 1.0 / K[:, 0, 0]
    intrinsics_inv[:, 1, 1] = 1.0 / K[:, 1, 1]
    intrinsics_inv[:, 0, 2] = -K[:, 0, 2] / K[:, 0, 0]
    intrinsics_inv[:, 1, 2] = -K[:, 1, 2] / K[:, 1, 1]
    homogeneous_coords = torch.cat(
        [pixel_coords, torch.ones_like(pixel_coords[:, :, :1])], dim=2
    )  # (H, W, 3)
    ray_directions = torch.matmul(
        intrinsics_inv, homogeneous_coords.permute(2, 0, 1).flatten(1)
    )  # (3, H*W)
    ray_directions = nn.functional.normalize(ray_directions, dim=1)  # (B, 3, H*W)
    ray_directions = ray_directions.permute(0, 2, 1)  # (B, H*W, 3)

    theta = torch.atan2(ray_directions[..., 0], ray_directions[..., -1])
    phi = torch.acos(ray_directions[..., 1])
    angles = torch.stack([theta, phi], dim=-1)
    return ray_directions, angles


def spherical_zbuffer_to_euclidean(spherical_tensor: Tensor) -> Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract zbuffer depth

    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    # =>
    # r = z / sin(phi) / cos(theta)
    # y = z / (sin(phi) / cos(phi)) / cos(theta)
    # x = z * sin(theta) / cos(theta)
    x = z * torch.tan(theta)
    y = z / torch.tan(phi) / torch.cos(theta)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


def spherical_to_euclidean(spherical_tensor: Tensor) -> Tensor:
    theta = spherical_tensor[..., 0]  # Extract polar angle
    phi = spherical_tensor[..., 1]  # Extract azimuthal angle
    r = spherical_tensor[..., 2]  # Extract radius
    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    x = r * torch.sin(phi) * torch.sin(theta)
    y = r * torch.cos(phi)
    z = r * torch.cos(theta) * torch.sin(phi)

    euclidean_tensor = torch.stack((x, y, z), dim=-1)
    return euclidean_tensor


def euclidean_to_spherical(spherical_tensor: Tensor) -> Tensor:
    x = spherical_tensor[..., 0]  # Extract polar angle
    y = spherical_tensor[..., 1]  # Extract azimuthal angle
    z = spherical_tensor[..., 2]  # Extract radius
    # y = r * cos(phi)
    # x = r * sin(phi) * sin(theta)
    # z = r * sin(phi) * cos(theta)
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(x / r, z / r)
    phi = torch.acos(y / r)

    euclidean_tensor = torch.stack((theta, phi, r), dim=-1)
    return euclidean_tensor


def euclidean_to_spherical_zbuffer(euclidean_tensor: Tensor) -> Tensor:
    pitch = torch.asin(euclidean_tensor[..., 1])
    yaw = torch.atan2(euclidean_tensor[..., 0], euclidean_tensor[..., -1])
    z = euclidean_tensor[..., 2]  # Extract zbuffer depth
    euclidean_tensor = torch.stack((pitch, yaw, z), dim=-1)
    return euclidean_tensor


def unproject_points(depth: Tensor, camera_intrinsics: Tensor) -> Tensor:
    r"""
    Unprojects a batch of depth maps to 3D point clouds using camera intrinsics.

    Parameters
    ----------
    depth: Tensor[B, H, W]
        Batch of depth maps.
    camera_intrinsics: Tensor[B, 3, 3]
        Camera intrinsic matrix of shape.

    Returns
    -------
    Tensor[B, 3, H, W]
        Projected coordinates (XYZ) for each pixel in the depth map.
    """
    batch_size, height, width = depth.shape
    device = depth.device

    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    pixel_coords = torch.stack((x_coords, y_coords), dim=-1)  # (H, W, 2)

    # Get homogeneous coords (u v 1)
    pixel_coords_homogeneous = torch.cat(
        (pixel_coords, torch.ones((height, width, 1), device=device)), dim=-1
    )
    pixel_coords_homogeneous = pixel_coords_homogeneous.permute(2, 0, 1).flatten(
        1
    )  # (3, H*W)
    # Apply K^-1 @ (u v 1): [B, 3, 3] @ [3, H*W] -> [B, 3, H*W]
    unprojected_points = torch.matmul(
        torch.inverse(camera_intrinsics), pixel_coords_homogeneous
    )  # (B, 3, H*W)
    unprojected_points = unprojected_points.view(
        batch_size, 3, height, width
    )  # (B, 3, H, W)
    unprojected_points = unprojected_points * depth.unsqueeze(-3)  # (B, 3, H, W)
    return unprojected_points


def project_points(
    points_3d: Tensor,
    intrinsic_matrix: Tensor,
    image_shape: T.Tuple[int, int],
) -> Tensor:
    r"""
    Project 3D points onto the image plane using camera intrinsics:

    :math:`(u v w) = (x y z) @ K^T`

    Parameters
    ----------
    points_3d: Tensor[B, N, 3]
        Batch of 3D points.
    intrinsic_matrix: Tensor[B, 3, 3]
        Camera intrinsic matrix.
    image_shape: Tuple[int, int]
        Image shape (height, width).

    Returns
    -------
    Tensor[B, H, W]
        Depth map from the projected points.
    """
    points_2d = torch.matmul(points_3d, intrinsic_matrix.transpose(1, 2))

    # Normalize projected points: (u v w) -> (u / w, v / w, 1)
    points_2d = points_2d[..., :2] / points_2d[..., 2:]

    points_2d = points_2d.int()

    # Valid points inside the image plane
    valid_mask = (
        (points_2d[..., 0] >= 0)
        & (points_2d[..., 0] < image_shape[1])
        & (points_2d[..., 1] >= 0)
        & (points_2d[..., 1] < image_shape[0])
    )

    # Calculate the flat indices of the valid pixels
    flat_points_2d = points_2d[..., 0] + points_2d[..., 1] * image_shape[1]
    flat_indices = flat_points_2d.long()

    # Create depth maps and counts using scatter_add, (B, H, W)
    depth_maps = torch.zeros(
        [points_3d.shape[0], *image_shape], device=points_3d.device
    )
    counts = torch.zeros([points_3d.shape[0], *image_shape], device=points_3d.device)

    # Loop over batches to apply masks and accumulate depth/count values
    for i in range(points_3d.shape[0]):
        valid_indices = flat_indices[i, valid_mask[i]]
        depth_maps[i].view(-1).scatter_add_(
            0, valid_indices, points_3d[i, valid_mask[i], 2]
        )
        counts[i].view(-1).scatter_add_(
            0, valid_indices, torch.ones_like(points_3d[i, valid_mask[i], 2])
        )

    # Calculate mean depth for each pixel in each batch
    mean_depth_maps = depth_maps / counts.clamp(min=1.0)
    return mean_depth_maps.reshape(-1, 1, *image_shape)  # (B, 1, H, W)


def downsample(data: Tensor, factor: int):
    """
    Downsample the input data tensor by taking the minimum value of each
    factor x factor block.

    Parameters
    ----------
    data: Tensor[B, C, H, W]
        Input data tensor.
    factor: int
        Downsampling factor.

    Returns
    -------
    Tensor[B, C, H // factor, W // factor]
        Downsampled data tensor.
    """
    N, _, H, W = data.shape
    data = data.view(
        N,
        H // factor,
        factor,
        W // factor,
        factor,
        1,
    )
    data = data.permute(0, 1, 3, 5, 2, 4).contiguous()
    data = data.view(-1, factor * factor)
    data_tmp = torch.where(data <= 0, torch.inf, data)
    data = torch.min(data_tmp, dim=-1).values
    data = data.view(N, 1, H // factor, W // factor)
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data


def flat_interpolate(
    flat_tensor: Tensor,
    shape_cur: T.Tuple[int, int],
    shape_new: T.Tuple[int, int],
    antialias: bool = True,
    mode: str = "bilinear",
) -> Tensor:
    """
    Interpolates a flat tensor of shape (B, C, H * W) to a new shape (B, C, H * W).

    Parameters
    ----------
    flat_tensor: Tensor[B, C, H * W]
        Input flat tensor.
    shape_cur: Tuple[int, int]
        Current shape of the tensor.
    shape_new: Tuple[int, int]
        New shape of the tensor.
    antialias: bool
        Whether to use antialiasing.
    mode: str
        Interpolation mode.

    Returns
    -------
    Tensor[B, C, H * W]
        Interpolated flat tensor.
    """

    if shape_cur == shape_new:
        return flat_tensor

    tensor = flat_tensor.view(
        flat_tensor.shape[0], shape_cur[0], shape_cur[1], -1
    ).permute(
        0, 3, 1, 2
    )  # b c h w
    tensor_interp = nn.functional.interpolate(
        tensor,
        size=(shape_new[0], shape_new[1]),
        mode=mode,
        align_corners=False,
        antialias=antialias,
    )
    flat_tensor_interp = tensor_interp.view(
        flat_tensor.shape[0], -1, shape_new[0] * shape_new[1]
    ).permute(
        0, 2, 1
    )  # b (h w) c
    return flat_tensor_interp.contiguous()
