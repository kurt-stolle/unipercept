r"""
Implements the camera s a TVTensor.
"""

from __future__ import annotations

import typing as T

import torch
import typing_extensions as TX
from unipercept.types import Size, Tensor
from torchvision.transforms.v2.functional import register_kernel
from torchvision.transforms.v2.functional._geometry import _compute_resized_output_size
from torchvision.tv_tensors import TVTensor

from unipercept.vision.geometry import (
    AxesConvention,
    euclidean_to_homogeneous_points,
    extrinsics_from_parameters,
    homogeneous_to_euclidean_points,
    intrinsics_from_parameters,
    unsafe_inverse,
    apply_points,
    generate_coord_grid,
)

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
            focal_length, principal_point, orthographic=False  # type: ignore
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
            spatial_size = canvas.flip(-1)  # H, W -> W, H
            canvas = torch.zeros((*intrinsics.shape[:-2], 4, 2), dtype=intrinsics.dtype)
            canvas[..., 2:, 0] = spatial_size  # canvas bbox end
            canvas[..., 2:, 1] = spatial_size  # image bbox end
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
    def I(self) -> Tensor:
        """
        Returns the intrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return self[..., :, 0:4]

    @property
    def I_inv(self) -> Tensor:
        """
        Returns the inverse intrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return unsafe_inverse(self.I)

    @property
    def E(self) -> Tensor:
        """
        Returns the extrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return self[..., :, 4:8]

    @property
    def E_inv(self) -> Tensor:
        """
        Returns the inverse extrinsic matrix as a (batch_size, 4, 4) tensor.
        """
        return unsafe_inverse(self.E)

    @property
    def crop_bbox(self) -> Tensor:
        """
        Returns the image crop bounding box as a [B, (u1, v1, u2, v2)] tensor.
        """
        return self[..., 8]

    @property
    def crop_size(self) -> Tensor:
        """
        Returns the image size as a (batch_size, (V, U)) tensor.
        """
        bbox = self.crop_bbox
        size = bbox[..., 2:] - bbox[..., :2]
        return size.flip(-1)

    @property
    def crop_height(self) -> Tensor:
        """
        Returns the image height as a (batch_size, V) tensor.
        """
        return self.crop_bbox[..., 3] - self.crop_bbox[..., 1]

    @property
    def crop_width(self) -> Tensor:
        """
        Returns the image width as a (batch_size, U) tensor.
        """
        return self.crop_bbox[..., 2] - self.crop_bbox[..., 0]

    @property
    def crop_center(self) -> Tensor:
        """
        Returns the image center as a (batch_size, (U, V)) tensor.
        """
        bbox = self.crop_bbox
        size = self.crop_size
        return bbox[..., :2] + size.flip(-1) / 2

    @property
    def canvas_bbox(self) -> Tensor:
        """
        Returns the canvas crop bounding box as a [B, (u1, v1, u2, v2)] tensor.
        """
        return self[..., 9]

    @property
    def canvas_size(self) -> Tensor:
        """
        Returns the canvas size as a (batch_size, (V, U)) tensor.
        """
        bbox = self.canvas_bbox
        size = bbox[..., 2:] - bbox[..., :2]
        return size.flip(-1)

    @property
    def canvas_height(self) -> Tensor:
        """
        Returns the canvas height as a (batch_size, V) tensor.
        """
        return self.canvas_bbox[..., 3] - self.canvas_bbox[..., 1]

    @property
    def canvas_width(self) -> Tensor:
        """
        Returns the canvas width as a (batch_size, U) tensor.
        """
        return self.canvas_bbox[..., 2] - self.canvas_bbox[..., 0]

    @property
    def canvas_center(self) -> Tensor:
        """
        Returns the canvas center as a (batch_size, (U, V)) tensor.
        """
        bbox = self.canvas_bbox
        size = self.canvas_size
        return bbox[..., :2] + size.flip(-1) / 2

    ##########################
    # Sub-matrices and views #
    ##########################

    @property
    def focal_length(self) -> Tensor:
        """
        Returns the focal length of the camera as a [..., (fx, fy)] tensor.
        """
        return self[..., :2, :2].diagonal(0, -2, -1)

    @property
    def principal_point(self) -> Tensor:
        """
        Returns the principal point of the camera as a [..., (u0, v0)] tensor.
        """
        return self[..., :2, 2]

    @property
    def translation(self) -> Tensor:
        """
        Returns the translation part of the extrinsic parameters as a (batch_size, 3) tensor.
        """
        return self[..., :3, 4:7]

    @property
    def K(self) -> Tensor:
        """
        Returns the camera matrix of the camera.
        """
        return self[..., :3, 0:3]

    @property
    def R(self) -> Tensor:
        """
        Returns the rotation part of the extrinsic parameters as a (batch_size, 3, 3) tensor.
        """
        return self[..., :3, 4:7]

    @property
    def Rt(self) -> Tensor:
        """
        Rotation-translation matrix, i.e. the concatenation :math:`[R|t]`.

        Returns
        -------
        Tensor[..., 3, 4]
            Rotation-translation matrix.
        """
        return self[..., :3, 4:8]

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

    @torch.no_grad()
    def get_fov_map(self, like: Size | Tensor) -> Tensor:
        r"""
        Encode the camera intrinsic parameters (focal length and principle point) to a map
        with channels for the distance to the principle point and the field of view.
        """
        if isinstance(like, Tensor):
            height, width = like.shape[-2:]
        else:
            height, width = like[-2:]

        shape = self.shape[:-2]
        if len(shape) == 0:
            self = self.unsqueeze(0).as_subclass(PinholeCamera)
        elif len(shape) > 1:
            self = self.flatten(0, -3).as_subclass(PinholeCamera)

        results = []
        for focal_length, principal_point in zip(
            self.focal_length, self.principal_point
        ):
            fx, fy = focal_length.unbind(-1)
            u0, v0 = principal_point.unbind(-1)

            # Distance to principal point
            x_row = torch.arange(0, width, dtype=torch.float32)
            x_row_center_norm = (x_row - u0) / (width * 2)
            x_center = torch.tile(x_row_center_norm, (height, 1))  # [H, W]

            y_col = torch.arange(0, height, dtype=torch.float32)
            y_col_center_norm = (y_col - v0) / (height * 2)
            y_center = torch.tile(y_col_center_norm, (width, 1)).T

            # FoV
            fov_x = torch.arctan(x_center / (fx / width))
            fov_y = torch.arctan(y_center / (fy / height))

            res = torch.stack([x_center, y_center, fov_x, fov_y], dim=2)
            results.append(res)

        res = torch.stack(results)

        if len(shape) == 0:
            res = res.squeeze(0)
        elif len(shape) > 1:
            res = res.unflatten(0, shape)

        return res

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

        *batch_shape, height, width = depth_map.shape

        # Create pixel grid
        point_coords = generate_coord_grid(
            (height, width),
            device=depth_map.device,
            dtype=depth_map.dtype,
            noise=noise,
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
        image_shape: T.Tuple[int, int] | Size | T.Sequence[int],
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
        points_2d = torch.matmul(points, self.K.transpose(1, 2))

        # Normalize projected points: (u v w) -> (u / w, v / w, 1)
        points_2d = points_2d[..., :2] / points_2d[..., 2:]
        points_2d = points_2d.int()

        # Valid points inside the image plane
        x_min, y_min, x_max, y_max = self.crop_bbox
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
        return mean_depth_maps.reshape(-1, *image_shape)  # (B, H, W)

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
        factor_wh = factor_hw.flip(-1)

        # Modify the intrinsic matrix
        self.principal_point[..., :] *= factor_wh
        self.focal_length[..., :] *= factor_hw

        # Modify the image size
        crop_center = self.crop_center * factor_wh
        crop_size = (self.crop_size * factor_hw).flip(-1)

        self.crop_bbox[:] = torch.cat(
            [crop_center - crop_size / 2, crop_center + crop_size / 2], dim=-1
        )

        # Modify the crop (and convert H, W -> V, U)
        canvas_center = self.canvas_center * factor_wh
        canvas_size = (self.canvas_size * factor_hw).flip(-1)

        self.crop_bbox[:] = torch.cat(
            [canvas_center - canvas_size / 2, canvas_center + canvas_size / 2], dim=-1
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
                "canvas_center": self.canvas_center.tolist(),
                "crop_size": self.crop_size.tolist(),
                "crop_center": self.crop_center.tolist(),
            }.items()
        )
        return f"{self.__class__.__name__}({self.dtype}, {kwds}){{\n{TAB}{fields}\n}}"


#################################################
# Functions for extracting parameters           #
# without calling `.as_subclass(PinholeCamera)` #
#################################################


def get_focal_length(cam: Tensor) -> Tensor:
    return cam[..., :2, :2].diagonal(0, -2, -1)


def get_principal_point(cam: Tensor) -> Tensor:
    return cam[..., :2, 2]


def get_translation(cam: Tensor) -> Tensor:
    return cam[..., :3, 4:7]


def get_camera_matrix(cam: Tensor) -> Tensor:
    return cam[..., :3, :3]


def get_rotation_matrix(cam: Tensor) -> Tensor:
    return cam[..., :3, 4:7]


def get_intrinsics(cam: Tensor) -> Tensor:
    return cam[..., :4, :4]


def get_extrinsics(cam: Tensor) -> Tensor:
    return cam[..., :4, 4:8]


def get_image_bbox(cam: Tensor) -> Tensor:
    return cam[..., 9]


def get_crop_bbox(cam: Tensor) -> Tensor:
    return cam[..., 8]


#################################
# Register Torchvision handlers #
#################################


@register_kernel(functional="resize", tv_tensor_cls=PinholeCamera)
def resize_camera(
    camera: PinholeCamera,
    size: T.List[int],
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
    crop = torch.tensor([left, top, left + width, top + height])
    camera = camera.crop(crop)
    return camera.as_subclass(PinholeCamera)
