r"""
Implements the camera s a TVTensor.

By convention, all spatial shapes are (H, W) and coordinates are (X, Y) where X is the wide axis and Y is the tall axis.
"""

from __future__ import annotations

import functools
import typing as T

import torch
import typing_extensions as TX
from torchvision.transforms.v2.functional import register_kernel
from torchvision.transforms.v2.functional._geometry import _compute_resized_output_size
from torchvision.tv_tensors import TVTensor

from unipercept.types import Size, Tensor
from unipercept.vision.coord import GridMode, generate_coord_grid
from unipercept.vision.geometry import (
    AxesConvention,
    apply_points,
    euclidean_to_homogeneous_points,
    extrinsics_from_parameters,
    generate_fovmap,
    homogeneous_to_euclidean_points,
    intrinsics_from_parameters,
    unsafe_inverse,
)

__all__ = [
    "PinholeCamera",
    "get_intrinsics",
    "get_focal_length",
    "get_principal_point",
    "get_extrinsics",
    "get_canvas_size",
    "get_sensor_size",
    "get_crop_bbox",
]

#####################
# Utility functions #
#####################


def _flip_spatial(tensor: Tensor) -> Tensor:
    r"""
    Flips the final (spatial) dimensions of a tensor without copying the data.
    """
    # assert_shape(tensor, (..., 2))
    return tensor[..., (1, 0)]


_P = T.ParamSpec("_P")


def _as_tensor(wrap: T.Callable[_P, Tensor]) -> T.Callable[_P, Tensor]:
    r"""
    Converts the input to a tensor.
    """

    @functools.wraps(wrap)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Tensor:
        sub = wrap(*args, **kwargs)
        if torch.compiler.is_compiling():
            return sub
        return sub.as_subclass(torch.Tensor)

    return wrapper  # type: ignore


####################
# Camera container #
####################

INDEX_INTRINSICS = (..., slice(0, 4), slice(0, 4))
INDEX_EXTRINSICS = (..., slice(0, 4), slice(4, 8))
INDEX_CROP_BBOX = (..., slice(0, 4), 8)
INDEX_CANVAS_SIZE = (..., (0, 1), 9)
INDEX_SENSOR_SIZE = (..., (2, 3), 9)


class PinholeCamera(TVTensor):
    """
    A PinholeCamera is a TVTensor that represents a pinhole camera model.
    This tensor always has a shape of [..., 4, 9], where the last dimension
    represents a concatenation of the intrinsic and extrinsic parameters of the camera,
    plus an additional column that represents the image size:

    - [..., :4, 0:4] represents the intrinsic matrix
    - [..., :4, 4:8] represents the extrinsic matrix
    - [..., :4, 8  ] represents the image crop bounding box as (u1, v1, u2, v2) with respect to the image size
    - [..., :2, 9  ] represents the (transformed) image size as (U, V) without cropping applied
    - [..., 2:, 9  ] represents the (original) sensor size as (U, V)
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
        F = torch.tensor([1.0, 1.0]) * max(H, W) * 2
        P = torch.tensor([W / 2, H / 2])

        cam = cls.from_parameters(F, P, canvas=[W, H])
        cam = cam.unsqueeze(0)
        cam = cam.expand((n_cameras, -1, -1))
        return cam.as_subclass(cls)

    @classmethod
    def from_parameters(
        cls,
        focal_length: Tensor | T.Iterable[int | float],
        principal_point: Tensor | T.Iterable[int | float],
        angles: Tensor | T.Iterable[int | float] | None = None,  # pitch, yaw, roll
        translation: Tensor | T.Iterable[int | float] | None = None,  # x, y, z
        convention: AxesConvention | str = AxesConvention.ISO8855,
        canvas: Tensor | T.Iterable[int] | None = None,
    ) -> T.Self:
        """
        Constructs a PinholeCamera tensor from the camera parameters.
        """
        focal_length = torch.as_tensor(focal_length).float()
        principal_point = torch.as_tensor(principal_point).to(focal_length)

        intrinsics = intrinsics_from_parameters(
            focal_length,
            principal_point,
            orthographic=False,  # type: ignore
        )

        if angles is None and translation is None:
            extrinsics = None
        else:
            if angles is None:
                translation = torch.as_tensor(translation).to(focal_length)
                angles = torch.zeros_like(translation)
            else:
                angles = torch.as_tensor(angles).to(focal_length)
                translation = torch.as_tensor(translation)
            extrinsics = extrinsics_from_parameters(angles, translation, convention)

        if canvas is not None:
            canvas = torch.as_tensor(canvas).to(focal_length)

        return cls.from_parts(intrinsics, extrinsics, canvas)

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
            extrinsics[..., :, :] = torch.eye(4, device=intrinsics.device)
        else:
            extrinsics = torch.as_tensor(extrinsics)
        assert extrinsics.ndim >= 2 and extrinsics.size(-1) == 4

        if canvas is None:
            u0, v0 = intrinsics[..., 0, 2], intrinsics[..., 1, 2]
            canvas = torch.cat([v0 * 2, u0 * 2], dim=-1)
        else:
            canvas = torch.as_tensor(canvas).to(intrinsics)
        if canvas.size(-1) == 2:
            spatial_size = _flip_spatial(canvas)  # H, W -> W, H
            canvas = torch.zeros((*intrinsics.shape[:-2], 4, 2), dtype=intrinsics.dtype)
            canvas[..., 2:, 0] = spatial_size  # canvas bbox end
            canvas[..., :2, 1] = spatial_size  # image size
            canvas[..., 2:, 1] = spatial_size  # sensor size
        elif canvas.size(-1) != 4 or canvas.ndim != intrinsics.ndim:
            msg = (
                "Canvas must be a tensor of shape [(V, U)] or "
                "[(img_left, img_top, img_right, img_bottom), "
                "(crop_left, crop_top, crop_right, crop_bottom)]. "
                f"Got: {canvas.shape}"
            )
            raise ValueError(msg)

        res = torch.zeros((*intrinsics.shape[:-2], 4, 10), dtype=intrinsics.dtype)
        res[..., :, :4] = intrinsics
        res[..., :, 4:8] = extrinsics
        res[..., :, 8:] = canvas
        return res.as_subclass(cls)

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
        return tensor.size(-2) == 4 and tensor.size(-1) == 10

    ########################
    # Tensor destructuring #
    ########################

    @property
    # @_as_tensor
    def I(self) -> Tensor:
        """
        Returns the intrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return get_intrinsics(self)

    @property
    # @_as_tensor
    def I_inv(self) -> Tensor:
        """
        Returns the inverse intrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return unsafe_inverse(self.I)

    @property
    # @_as_tensor
    def E(self) -> Tensor:
        """
        Returns the extrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return get_extrinsics(self)

    @property
    # @_as_tensor
    def E_inv(self) -> Tensor:
        """
        Returns the inverse extrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return unsafe_inverse(self.E)

    @property
    # @_as_tensor
    def crop_bbox(self) -> Tensor:
        """
        Returns the image crop bounding box as a [B, (u1, v1, u2, v2)] tensor.
        """
        return get_crop_bbox(self)

    @property
    # @_as_tensor
    def crop_size(self) -> Tensor:
        """
        Returns the image size as a (batch_size, (V, U)) tensor.
        """
        return get_crop_size(self)

    @property
    # @_as_tensor
    def crop_height(self) -> Tensor:
        """
        Returns the image height as a (batch_size, V) tensor.
        """
        return get_crop_size(self)[..., 0]

    @property
    # @_as_tensor
    def crop_width(self) -> Tensor:
        """
        Returns the image width as a (batch_size, U) tensor.
        """
        return get_crop_size(self)[..., 1]

    @property
    # @_as_tensor
    def crop_center(self) -> Tensor:
        """
        Returns the image center as a (batch_size, (U, V)) tensor.
        """
        bbox = self.crop_bbox
        size = self.crop_size
        center = bbox[..., :2] + _flip_spatial(size) / 2
        return center

    @property
    # @_as_tensor
    def canvas_size(self) -> Tensor:
        """
        Returns the canvas size as a (batch_size, (V, U)) tensor.
        """
        return get_canvas_size(self)

    @property
    # @_as_tensor
    def canvas_height(self) -> Tensor:
        """
        Returns the canvas height as a (batch_size, V) tensor.
        """
        return get_canvas_size(self)[..., 0]

    @property
    # @_as_tensor
    def canvas_width(self) -> Tensor:
        """
        Returns the canvas width as a (batch_size, U) tensor.
        """
        return get_canvas_size(self)[..., 1]

    @property
    # @_as_tensor
    def sensor_size(self) -> Tensor:
        return get_sensor_size(self)

    @property
    # @_as_tensor
    def sensor_height(self) -> Tensor:
        return self.sensor_size[..., 0]

    @property
    # @_as_tensor
    def sensor_width(self) -> Tensor:
        return self.sensor_size[..., 1]

    ##########################
    # Sub-matrices and views #
    ##########################

    @property
    def focal_length(self) -> Tensor:
        """
        Returns the focal length of the camera as a [..., (fx, fy)] tensor.
        """
        return get_focal_length(self)

    @property
    def principal_point(self) -> Tensor:
        """
        Returns the principal point of the camera as a [..., (u0, v0)] tensor.
        """
        return get_principal_point(self)

    @property
    def translation(self) -> Tensor:
        """
        Returns the translation part of the extrinsic parameters as a (batch_size, 3) tensor.
        """
        return get_translation(self)

    @property
    def K(self) -> Tensor:
        """
        Returns the camera matrix of the camera.
        """
        return get_intrinsics(self)[..., :3, :3]

    @property
    def R(self) -> Tensor:
        """
        Returns the rotation part of the extrinsic parameters as a (batch_size, 3, 3) tensor.
        """
        return get_extrinsics(self)[..., :3, :3]

    @property
    def Rt(self) -> Tensor:
        """
        Rotation-translation matrix, i.e. the concatenation :math:`[R|t]`.

        Returns
        -------
        Tensor[..., 3, 4]
            Rotation-translation matrix.
        """
        return get_extrinsics(self)[..., :3, :]

    @property
    def P(self) -> Tensor:
        """
        Projection matrix, defined as :math:`P = IE`.

        Returns
        -------
        Tensor[..., 4, 4]
            Projection matrix.
        """
        return self.I @ self.E

    @property
    def P_inv(self) -> Tensor:
        """
        Inverse projection matrix :math:`P^{-1}`.

        Returns
        -------
        Tensor[..., 4, 4]
            Inverse projection matrix.
        """
        return unsafe_inverse(self.P)

    ##########################
    # Representation methods #
    ##########################

    def generate_fovmap(self, like: Size | Tensor) -> Tensor:
        return generate_fovmap(self.focal_length, self.principal_point, like)

    def normalize_points(self, points_2d: Tensor) -> Tensor:
        r"""
        Normalize a set of 2D points to the camera plane.

        Returns
        -------
        Tensor[B, N, 2]
            Normalized points in the camera plane.
        """
        return (points_2d - self.principal_point) / self.focal_length

    def denormalize_points(self, points_2d: Tensor) -> Tensor:
        r"""
        Denormalize a set of 2D points to the image plane.

        Returns
        -------
        Tensor[B, N, 2]
            Denormalized points in the image plane.
        """
        return points_2d * self.focal_length + self.principal_point

    def reproject_map(self, depth_map: Tensor, noise: bool = False) -> Tensor:
        r"""
        reprojects each pixel to 3D coordinates.

        Parameters
        ----------
        depth_map: Tensor[..., H, W]
            Depth map containing depth values for every pixel.
        noise:
            Whether to add noise to the pixel coordinates, see :func:`unipercept.vision.geometry.generate_coord_grid`.

        Returns
        -------
        Tensor[B, H, W, 3]
            Projected coordinates (XYZ) for each pixel in the depth map.
        """
        assert depth_map.ndim >= 3
        depth_map = depth_map.as_subclass(torch.Tensor)

        *batch_shape, height, width = depth_map.shape

        # Create pixel grid
        point_coords = generate_coord_grid(
            (height, width),
            device=depth_map.device,
            dtype=depth_map.dtype,
            mode=GridMode.PIXEL_NOISE if noise else GridMode.PIXEL_CENTER,
        )  # (H, W, 2)
        point_coords = point_coords.flatten(0, 1)  # (H*W, 2)

        # Reshape depth map to a depth value for every point
        point_depths = depth_map.flatten(0, -3)  # (B', H, W)
        point_depths = point_depths.reshape(-1, height * width, 1)  # (B', H*W, 1)

        # Repeat pixel coordinates for each batch item
        point_coords = point_coords.unsqueeze(0).expand(
            point_depths.size(0), -1, -1
        )  # (B', H*W, 2)

        result = self.image_to_world(point_coords, point_depths)  # (B', H*W, 3)
        result = result.unflatten(1, (height, width))
        result = result.unflatten(0, batch_shape)

        return result

    def image_to_camera(
        self,
        coords: Tensor,
        depths: Tensor | None = None,
    ) -> Tensor:
        r"""
        Reprojects pixel coordinates to the camera plane. If depth is None, then
        the pixels are projected to the z=1 plane.
        """
        # points = self.normalize_points(coords)
        # points = nn.functional.normalize(points, dim=-1, p=2.0)
        # if depths is not None:
        #    points = points * depths
        points = euclidean_to_homogeneous_points(coords)
        points = apply_points(self.I_inv, points) * depths
        return points

    def camera_to_world(
        self,
        points: Tensor,
    ) -> Tensor:
        r"""
        Reprojects a set of 3D points in the camera system to 3D points in the world.

        Parameters
        ----------
        points: Tensor[B, N, 3]
            2D points to reproject.

        Returns
        -------
        Tensor[B, N, 3]
            Projected coordinates (XYZ) for each pixel in the depth map.
        """
        points = apply_points(self.E_inv, points)
        return points

    def image_to_world(
        self,
        coords: Tensor,
        depths: Tensor | None = None,
    ) -> Tensor:
        r"""
        Reprojects pixel coordinates (u,v) to 3D coordinates (x,y,z) in the world.
        """
        points = self.image_to_camera(coords, depths)
        return self.camera_to_world(points)

    def project_map(
        self,
        points: Tensor,
        image_shape: tuple[int, int] | Size | T.Sequence[int],
    ) -> Tensor:
        r"""
        Project a set of points onto a 2D depth map.

        Parameters
        ----------
        points: Tensor[B, N, 3]
            Batch of 3D points.
        image_shape: Tuple[int, int]
            Image shape (height, width).

        Returns
        -------
        Tensor[B, H, W]
            Depth map from the projected points.
        """
        shape = points.shape
        points = points.reshape(-1, *shape[-2:]).contiguous()

        points = apply_points(self.E, points)  # (x y z)
        points_2d = apply_points(self.I, points)  # (u v w)

        # Normalize projected points: (u v w) -> (u / w, v / w, 1)
        # points_2d = points_2d[..., :2] * points_2d[..., 2:]

        points_2d = points_2d.int()
        # Valid points inside the image plane
        # x_min, y_min, x_max, y_max = self.crop_bbox.unbind(-1)
        x_min, y_min, x_max, y_max = 0, 0, image_shape[1], image_shape[0]
        valid_mask = (
            (points_2d[..., 0] >= x_min)
            & (points_2d[..., 0] < x_max)
            & (points_2d[..., 1] >= y_min)
            & (points_2d[..., 1] < y_max)
        )

        # Calculate the flat indices of the valid pixels
        flat_points_2d = points_2d[..., 0] + points_2d[..., 1] * image_shape[1]
        flat_indices = flat_points_2d.long()

        # Create depth maps and counts using scatter_add, (B, H, W)
        depth_maps = torch.zeros([points.shape[0], *image_shape], device=points.device)
        counts = torch.zeros([points.shape[0], *image_shape], device=points.device)

        # Loop over batches to apply masks and accumulate depth/count values
        for i in range(points.shape[0]):
            valid_indices = flat_indices[i, valid_mask[i]]
            depth_maps[i].view(-1).scatter_add_(
                0, valid_indices, points[i, valid_mask[i], 2]
            )
            counts[i].view(-1).scatter_add_(
                0, valid_indices, torch.ones_like(points[i, valid_mask[i], 2])
            )

        # Calculate mean depth for each pixel in each batch
        mean_depth_maps = depth_maps / counts.clamp(min=1.0)
        return mean_depth_maps.reshape(*shape[:-2], *image_shape)

    def project_points(self, points: Tensor) -> Tensor:
        r"""
        Project a set of 3D points to 2D points.

        Parameters
        ----------
        points: Tensor[B, N, 3]
            3D points to project.

        Returns
        -------
        Tensor[B, N, 2]
            2D points projected by the camera.
        """
        result_homo = apply_points(self.P, points)
        return homogeneous_to_euclidean_points(result_homo)

    #####################
    # Modifying methods #
    #####################

    @torch.no_grad()
    def scale_(self, factor: Tensor) -> T.Self:
        """
        This is the inplace variant of :meth:`scale`.
        """
        if factor.size(-1) == 1:
            factor = factor.expand(*factor.shape[:-1], 2)
        elif factor.size(-1) != 2:
            msg = f"Expected a tensor of shape [..., (V, U)], but got {factor.shape}"
            raise ValueError(msg)

        factor_hw = factor.to(self)
        factor_wh = _flip_spatial(factor_hw)

        # Modify the intrinsic matrix
        self.principal_point[..., :] *= factor_wh
        self.focal_length[..., :] *= factor_hw

        # Modify the image size
        crop_center = self.crop_center * factor_wh
        crop_size = _flip_spatial(self.crop_size * factor_hw)

        self[INDEX_CROP_BBOX] = torch.cat(
            [crop_center - crop_size / 2, crop_center + crop_size / 2], dim=-1
        )

        # Modify the crop (and convert H, W -> V, U)
        self[INDEX_CANVAS_SIZE] *= factor_wh

        return self

    def scale(self, factor: Tensor, inplace: bool = False) -> T.Self:
        """
        Scales the camera by the given factor.

        Assumes the scaling is done w.r.t. the center of the current image crop.
        """
        if not inplace:
            self = self.clone().as_subclass(type(self))
        return self.scale_(factor)

    @torch.no_grad()
    def crop_(self, crop: Tensor) -> T.Self:
        """
        This is the inplace variant of :meth:`crop`.
        """
        crop_width = crop[..., 2] - crop[..., 0]
        crop_height = crop[..., 3] - crop[..., 1]

        # Move the principal point
        self.principal_point[..., :] -= crop[..., :2]

        assert (
            self.canvas_width >= crop_width
        ), f"Cannot crop outside bounds: {self.canvas_width=} < {crop_width=}"
        assert (
            self.canvas_height >= crop_height
        ), f"Cannot crop outside bounds: {self.canvas_height=} < {crop_height=}"

        self.crop_bbox[:] = crop.to(self)
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

    def horizontal_flip_(self):
        """
        This is the inplace variant of :meth:`horizontal_flip`.
        """

        self.principal_point[..., 0] = self.crop_width - self.principal_point[..., 0]

        T = torch.eye(4, device=self.device, dtype=self.dtype)
        T[..., 0, 0] = -1
        self.E[:] = self.E @ T

        return self

    def horizontal_flip(self, inplace: bool = False):
        """
        Flips the camera horizontally.

        This effectively mirrors the image along the vertical axis.
        """
        if not inplace:
            self = self.clone().as_subclass(type(self))
        return self.horizontal_flip_()

    def vertical_flip_(self):
        """
        This is the inplace variant of :meth:`vertical_flip`.
        """
        self.principal_point[..., 1] = self.crop_height - self.principal_point[..., 1]

        T = torch.eye(4, device=self.device, dtype=self.dtype)
        T[..., 1, 1] = -1
        self.E[:] = self.E @ T

        return self

    def vertial_flip(self, inplace: bool = False):
        """
        Flips the camera vertically.

        This effectively mirrors the image along the horizontal axis.
        """
        if not inplace:
            self = self.clone().as_subclass(type(self))
        return self.vertical_flip_()

    @TX.override
    def __repr__(self, *args, **kwargs):
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
                "rotation": self.R.tolist(),
                "translation": self.translation.tolist(),
                "canvas_size": self.canvas_size.tolist(),
                "sensor_size": self.sensor_size.tolist(),
                "crop_size": self.crop_size.tolist(),
                "crop_center": self.crop_center.tolist(),
            }.items()
        )
        return f"{self.__class__.__name__}({self.dtype}, {kwds}){{\n{TAB}{fields}\n}}"


#################################################
# Functions for extracting parameters           #
# without calling `.as_subclass(PinholeCamera)` #
#################################################


@_as_tensor
def get_intrinsics(cam: Tensor) -> Tensor:
    return cam[INDEX_INTRINSICS]


@_as_tensor
def get_focal_length(cam: Tensor) -> Tensor:
    return cam[..., :2, :2].diagonal(0, -2, -1)


@_as_tensor
def get_principal_point(cam: Tensor) -> Tensor:
    return cam[..., :2, 2]


@_as_tensor
def get_translation(cam: Tensor) -> Tensor:
    return cam[..., :3, 7]


@_as_tensor
def get_camera_matrix(cam: Tensor) -> Tensor:
    return cam[..., :3, :3]


@_as_tensor
def get_rotation_matrix(cam: Tensor) -> Tensor:
    return cam[..., :3, 4:7]


@_as_tensor
def get_extrinsics(cam: Tensor) -> Tensor:
    return cam[INDEX_EXTRINSICS]


@_as_tensor
def get_crop_bbox(cam: Tensor) -> Tensor:
    return cam[INDEX_CROP_BBOX]


@_as_tensor
def get_crop_size(cam: Tensor) -> Tensor:
    return _flip_spatial(get_crop_bbox(cam)[..., 2:] - get_crop_bbox(cam)[..., :2])


@_as_tensor
def get_canvas_size(cam: Tensor) -> Tensor:
    return _flip_spatial(cam[INDEX_CANVAS_SIZE])


@_as_tensor
def get_sensor_size(cam: Tensor) -> Tensor:
    return _flip_spatial(cam[INDEX_SENSOR_SIZE])


#################################
# Register Torchvision handlers #
#################################


@register_kernel(functional="resize", tv_tensor_cls=PinholeCamera)
def resize_camera(
    camera: PinholeCamera,
    size: list[int],
    interpolation: T.Any = None,  # noqa: U100
    max_size: int | None = None,
    antialias: T.Any = False,  # noqa: U100
    use_rescale: bool = False,
) -> Tensor:
    if camera.ndim >= 3:
        shape = camera.shape[:-2]
        cams = camera.flatten(0, -3)
        resized_cams = [
            resize_camera(
                cam.as_subclass(PinholeCamera),
                size,
                interpolation,
                max_size,
                antialias,
                use_rescale,
            )
            for cam in cams
        ]
        return torch.stack(resized_cams).unflatten(0, shape)

    h_old, w_old = camera.canvas_size.tolist()
    h_new, w_new = _compute_resized_output_size(
        (h_old, w_old), size=size, max_size=max_size
    )

    camera = camera.scale(torch.tensor([h_new / h_old, w_new / w_old]))
    return camera.as_subclass(PinholeCamera)


@register_kernel(functional="crop", tv_tensor_cls=PinholeCamera)
def crop_camera(
    camera: PinholeCamera,
    top: int,
    left: int,
    height: int,
    width: int,
) -> Tensor:
    crop = torch.tensor(
        [left, top, left + width, top + height],
        device=camera.device,
        dtype=camera.dtype,
    )
    camera = camera.crop(crop)
    return camera.as_subclass(PinholeCamera)


# TODO: flip
@register_kernel(functional="horizontal_flip", tv_tensor_cls=PinholeCamera)
def horizontal_flip_camera(
    camera: PinholeCamera,
) -> Tensor:
    return camera.horizontal_flip().as_subclass(PinholeCamera)


# TODO: pad
@register_kernel(functional="vertical_flip", tv_tensor_cls=PinholeCamera)
def vertical_flip_camera(
    camera: PinholeCamera,
) -> Tensor:
    return camera.vertial_flip().as_subclass(PinholeCamera)
